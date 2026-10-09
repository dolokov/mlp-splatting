// Opt-in TRAINING forward, grouped inside each tile/primitive (not global CSR).
// Included after rgb_wmma.cuh: reuse its unchanged v6 MMA and cast helpers.
// The original FP32 training record API remains the default and the fallback.
#include <cmath>
#include <string>

static std::atomic<int64_t> training_wmma_calls{0}, training_wmma_frame_fallbacks{0};
static std::atomic<int64_t> training_wmma_fp32_nonfinite_failures{0};
static std::atomic<int64_t> training_wmma_count_emit_failures{0};
static std::atomic<int64_t> training_wmma_capacity_failures{0};

// Process-lifetime counters, no reset. Calls are accepted count-pass launches;
// structural/input/candidate validation errors are excluded. Capacity failures
// count the per-pixel hit-capacity errors detected by the training kernel.
std::vector<int64_t> training_wmma_safety_stats() {
  return {training_wmma_calls.load(), training_wmma_frame_fallbacks.load(),
          training_wmma_fp32_nonfinite_failures.load(),
          training_wmma_count_emit_failures.load(), training_wmma_capacity_failures.load()};
}

enum TrainingWmmaStatus : int { TRAIN_WMMA_NUMERIC=1, TRAIN_WMMA_CAPACITY=2,
                              TRAIN_WMMA_COUNT_MISMATCH=4 };

// One fused read-only finite scan / one status readback. No casts or repairs.
struct TrainingWmmaFiniteInputs {
  const float* pointers[15];
  int64_t lengths[15];
  int count;
};

__global__ void training_wmma_finite_kernel(TrainingWmmaFiniteInputs inputs, int* status) {
  bool bad=false;
  for(int tensor=0;tensor<inputs.count;++tensor) {
    for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
        i<inputs.lengths[tensor];i+=(int64_t)blockDim.x*gridDim.x)
      bad |= !isfinite(inputs.pointers[tensor][i]);
  }
  if(bad) atomicOr(status,TRAIN_WMMA_NUMERIC);
}

static bool training_wmma_all_finite(const std::vector<torch::Tensor>& tensors) {
  TrainingWmmaFiniteInputs inputs{};
  int64_t largest=0;
  for(const auto& tensor:tensors) {
    if(tensor.scalar_type()!=at::kFloat) continue;
    TORCH_CHECK(inputs.count<15,"training WMMA finite scan has too many tensors");
    inputs.pointers[inputs.count]=tensor.data_ptr<float>();
    inputs.lengths[inputs.count]=tensor.numel();
    largest=std::max<int64_t>(largest,tensor.numel());
    ++inputs.count;
  }
  if(largest==0) return true;
  auto status=torch::zeros({},tensors[0].options().dtype(at::kInt));
  const int blocks=(int)std::min<int64_t>(4096,1+(largest-1)/256);
  training_wmma_finite_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
      inputs,status.data_ptr<int>());
  AT_CUDA_CHECK(cudaGetLastError());
  return status.item<int>()==0;
}

template<typename T>
__global__ __launch_bounds__(128,2) void render_training_wmma_kernel(
    const float* rays_o,const float* rays_d,const float* pos,const float* logs,const float* quats,
    const int32_t* candidates,const int32_t* tile_offsets,const float* freq,const float* att,
    const float* W1,const float* b1,const float* W2,const float* b2,const float* W3,const float* b3,
    int N,int width,int height,float threshold,float beta,float temperature,
    // Runtime emit flag: BOTH passes execute this exact compiled MLP/geometry
    // function. The count pass does not use a separate arithmetic specialization.
    bool emit,int32_t* counts,const int64_t* offsets,int64_t total_records,
    int32_t* ids,float* x6,float* raw,int32_t* masks,float* gates,
    float* rgb,float* out_alpha,int* status) {
  constexpr int threads=128,capacity=MLPSP_MAX_TILE_PRIMS,items=capacity/threads;
  constexpr int max_hits=MLPSP_TILECENTER_MAX_HITS;
  static_assert(capacity%threads==0,"Training WMMA sort needs a multiple of 128 candidates");
  using Sort=cub::BlockRadixSort<float,threads,items,int>;
  __shared__ int prims[capacity];
  static_assert(sizeof(typename Sort::TempStorage)<=sizeof(T)*(64*80+32*80),"sort scratch size");
  static_assert(sizeof(float)*64*32<=sizeof(T)*64*80,"activation alias size");
  __shared__ __align__(32) T staging[64*80+32*80];
  auto& sort_temp=*reinterpret_cast<typename Sort::TempStorage*>(staging);
  T* input=staging;
  T* weight1=staging+64*80;
  T* hidden=weight1;
  float* activation=reinterpret_cast<float*>(input);
  __shared__ __align__(32) T weight2[32*32],weight3[16*32];
  __shared__ float bias1[32],bias2[32],bias3[4];
  __shared__ bool active[64];
  __shared__ uint32_t actual_masks[64*2];
  const int tid=threadIdx.x,warp=tid/32,lane=tid&31;
  const int tile=blockIdx.y*(1+(width-1)/8)+blockIdx.x;
  const int x=blockIdx.x*8+tid%8,y=blockIdx.y*8+tid/8;
  const bool pixel=tid<64 && x<width && y<height;
  const int pix=pixel?y*width+x:0;
  const int64_t pixel3=(int64_t)pix*3;
  const int64_t center3=(int64_t)(min((int)blockIdx.y*8+4,height-1)*width+
                                min((int)blockIdx.x*8+4,width-1))*3;
  const float ox=pixel?rays_o[pixel3]:0.f,oy=pixel?rays_o[pixel3+1]:0.f,oz=pixel?rays_o[pixel3+2]:0.f;
  const float rx=pixel?rays_d[pixel3]:0.f,ry=pixel?rays_d[pixel3+1]:0.f,rz=pixel?rays_d[pixel3+2]:0.f;
  const int begin=tile_offsets[tile],length=tile_offsets[tile+1]-begin;
  bool bad=false;
  float keys[items];
  int values[items];
#pragma unroll
  for(int item=0;item<items;++item) {
    const int j=tid*items+item;
    if(j<length) {
      const int p=candidates[begin+j];
      const int64_t p3=(int64_t)p*3,p4=(int64_t)p*4;
      const float key=tstar_key_tilecenter(rays_o[center3],rays_o[center3+1],rays_o[center3+2],
          rays_d[center3],rays_d[center3+1],rays_d[center3+2],
          pos[p3],pos[p3+1],pos[p3+2],logs[p3],logs[p3+1],logs[p3+2],
          quats[p4],quats[p4+1],quats[p4+2],quats[p4+3]);
      bad |= !isfinite(key);
      keys[item]=key>0.f?key:1.e30f;
      values[item]=p;
    } else { keys[item]=1.e30f;values[item]=-1; }
  }
  // Same stable blocked-input order and FP32 key helper as scalar records.
  Sort(sort_temp).Sort(keys,values);
  __syncthreads();
#pragma unroll
  for(int item=0;item<items;++item) prims[tid*items+item]=values[item];
  __syncthreads(); // Sort storage is now available for WMMA staging.

  int used=0;
  float trans=1.f;
  float3 color=make_float3(0.f,0.f,0.f);
  int64_t record_begin=0,record_end=0;
  int expected_count=0;
  bool write_range_valid=true;
  if(emit && pixel) {
    record_begin=offsets[pix];record_end=offsets[pix+1];expected_count=counts[pix];
    write_range_valid=record_begin>=0 && record_end>=record_begin && record_end<=total_records &&
                      expected_count>=0 && expected_count<=max_hits && record_end-record_begin==expected_count;
    if(!write_range_valid) atomicOr(status,TRAIN_WMMA_COUNT_MISMATCH);
  }
  for(int j=0;j<length;++j) {
    const int p=prims[j];
    if(p<0 || p>=N) continue; // Uniform across this CTA; wrapper validates IDs.
    const int64_t p3=(int64_t)p*3,p4=(int64_t)p*4;
    float t=0.f,dx=0.f,dy=0.f,dz=0.f,qlx=0.f,qly=0.f,qlz=0.f,decay=0.f;
    bool hit=false;
    if(pixel && trans>=1.e-3f && used<max_hits) {
      hit=hit_and_features_rot(ox,oy,oz,rx,ry,rz,pos[p3],pos[p3+1],pos[p3+2],
          logs[p3],logs[p3+1],logs[p3+2],quats[p4],quats[p4+1],quats[p4+2],quats[p4+3],
          threshold,t,dx,dy,dz,qlx,qly,qlz,decay);
      bad |= !isfinite(t)||!isfinite(dx)||!isfinite(dy)||!isfinite(dz)||
             !isfinite(qlx)||!isfinite(qly)||!isfinite(qlz)||!isfinite(decay);
    }
    if(!__syncthreads_or(hit)) continue;
    if(tid<64) {
      active[tid]=hit;
      const float v[6]={rx,ry,rz,dx,dy,dz};
#pragma unroll
      for(int c=0;c<6;++c) {
#pragma unroll
        for(int f=0;f<6;++f) {
          const float phase=v[c]*freq[f]*att[f];
          bad |= hit && !isfinite(phase);
          input[tid*80+c*6+f]=rgb_mma_cast<T>(hit?__sinf(phase):0.f);
          input[tid*80+36+c*6+f]=rgb_mma_cast<T>(hit?__cosf(phase):0.f);
        }
      }
#pragma unroll
      for(int k=72;k<80;++k) input[tid*80+k]=rgb_mma_cast<T>(0.f);
    }
    for(int q=tid;q<32*80;q+=threads) {
      const int h=q/80,k=q%80;
      weight1[q]=rgb_mma_cast<T>(k<72?W1[(int64_t)p*2304+h*72+k]:0.f);
      bad |= !isfinite(rgb_mma_float(weight1[q]));
    }
    for(int q=tid;q<32*32;q+=threads) {
      weight2[q]=rgb_mma_cast<T>(W2[(int64_t)p*1024+q]);
      bad |= !isfinite(rgb_mma_float(weight2[q]));
    }
    for(int q=tid;q<16*32;q+=threads) {
      weight3[q]=rgb_mma_cast<T>(q<4*32?W3[(int64_t)p*128+q]:0.f);
      bad |= !isfinite(rgb_mma_float(weight3[q]));
    }
    if(tid<32) { bias1[tid]=b1[(int64_t)p*32+tid];bias2[tid]=b2[(int64_t)p*32+tid]; }
    if(tid<4) bias3[tid]=b3[p4+tid];
    __syncthreads();
    rgb_mma_first_layer_alias<T,80>(input,weight1,activation,warp);
    // ALL warps must finish consuming W1 before hidden overwrites it. The
    // first-layer helper also fences every input read before activation alias.
    __syncthreads();
    for(int row=warp*16;row<(warp+1)*16;++row) {
      const int q=row*32+lane;
      const float value=activation[q]+bias1[lane];
      const uint32_t mask=__ballot_sync(0xffffffffu,value>0.f);
      if(lane==0) actual_masks[row*2]=mask;
      hidden[q]=rgb_mma_cast<T>(fmaxf(value,0.f));
      bad |= active[row] && (!isfinite(value)||!isfinite(rgb_mma_float(hidden[q])));
    }
    __syncwarp(0xffffffffu);
    rgb_mma_layer<T,32,32>(hidden,weight2,activation,warp);
    __syncwarp(0xffffffffu);
    for(int row=warp*16;row<(warp+1)*16;++row) {
      const int q=row*32+lane;
      const float value=activation[q]+bias2[lane];
      const uint32_t mask=__ballot_sync(0xffffffffu,value>0.f);
      if(lane==0) actual_masks[row*2+1]=mask;
      hidden[q]=rgb_mma_cast<T>(fmaxf(value,0.f));
      bad |= active[row] && (!isfinite(value)||!isfinite(rgb_mma_float(hidden[q])));
    }
    __syncwarp(0xffffffffu);
    rgb_mma_layer<T,32,16>(hidden,weight3,activation,warp);
    __syncthreads(); // Original pixel owners read rows produced by other warps.
    if(hit) {
      const float r_logit=activation[tid*32]+bias3[0];
      const float g_logit=activation[tid*32+1]+bias3[1];
      const float b_logit=activation[tid*32+2]+bias3[2];
      const float sigma_logit=activation[tid*32+3]+bias3[3];
      const float R=fast_sigmoid(r_logit),G=fast_sigmoid(g_logit),B=fast_sigmoid(b_logit);
      const float sigma_raw=fast_softplus(sigma_logit);
      const float alpha=1.f-__expf(-sigma_raw*decay);
      float g0,g1,g2,g3;
      semantic_tetra_gates(qlx,qly,qlz,logs[p3],logs[p3+1],logs[p3+2],
                           beta,temperature,g0,g1,g2,g3);
      bad |= !isfinite(r_logit)||!isfinite(g_logit)||!isfinite(b_logit)||!isfinite(sigma_logit)||
             !isfinite(R)||!isfinite(G)||!isfinite(B)||!isfinite(sigma_raw)||!isfinite(alpha)||
             !isfinite(g0)||!isfinite(g1)||!isfinite(g2)||!isfinite(g3);
      if(emit) {
        // Never rely on count/emit agreement to make a write safe. Guard the
        // pixel's OWN interval as well as the total allocation before saving.
        if(write_range_valid && used<expected_count && used<record_end-record_begin) {
          const int64_t record=record_begin+used;
          ids[record]=p;
          x6[record*6]=rx;x6[record*6+1]=ry;x6[record*6+2]=rz;
          x6[record*6+3]=dx;x6[record*6+4]=dy;x6[record*6+5]=dz;
          masks[record*2]=(int32_t)actual_masks[tid*2];
          masks[record*2+1]=(int32_t)actual_masks[tid*2+1];
          raw[record*8]=R;raw[record*8+1]=G;raw[record*8+2]=B;raw[record*8+3]=sigma_raw;
          raw[record*8+4]=trans;raw[record*8+5]=alpha;raw[record*8+6]=decay;raw[record*8+7]=sigma_logit;
          gates[record*4]=g0;gates[record*4+1]=g1;gates[record*4+2]=g2;gates[record*4+3]=g3;
        } else atomicOr(status,TRAIN_WMMA_COUNT_MISMATCH);
      }
      const float weight=trans*alpha;
      color.x=fmaf(weight,R,color.x);color.y=fmaf(weight,G,color.y);color.z=fmaf(weight,B,color.z);
      trans*=1.f-alpha;
      ++used;
      bad |= !isfinite(trans)||!isfinite(color.x)||!isfinite(color.y)||!isfinite(color.z);
      // Match the scalar record contract even if this was the last candidate.
      if(trans>=1.e-3f && used>=max_hits) atomicOr(status,TRAIN_WMMA_CAPACITY);
    }
    if(!__syncthreads_or(pixel && trans>=1.e-3f)) break;
  }
  if(bad) atomicOr(status,TRAIN_WMMA_NUMERIC);
  if(pixel) {
    if(emit) {
      if(used!=expected_count) atomicOr(status,TRAIN_WMMA_COUNT_MISMATCH);
      rgb[pixel3]=color.x;rgb[pixel3+1]=color.y;rgb[pixel3+2]=color.z;
      out_alpha[pix]=1.f-trans;
    } else counts[pix]=used;
  }
}

static void training_wmma_check_structural_status(int status) {
  if(status & TRAIN_WMMA_CAPACITY) {
    training_wmma_capacity_failures.fetch_add(1);
    TORCH_CHECK(false,"Training WMMA hit capacity ",MLPSP_TILECENTER_MAX_HITS,
                " exceeded; refusing truncated records");
  }
  if(status & TRAIN_WMMA_COUNT_MISMATCH) {
    training_wmma_count_emit_failures.fetch_add(1);
    TORCH_CHECK(false,"Training WMMA count/emit mismatch; bounded partial records were discarded");
  }
}

std::vector<torch::Tensor> render_tiles_training_records_wmma_wrapper(
    const std::vector<torch::Tensor>& x,int64_t width,int64_t height,int64_t tile,
    double threshold,double beta,double temperature,const std::string& precision) {
  TORCH_CHECK(x.size()==17,"Training WMMA requires 15 scene inputs and two RGB/T adjoint placeholders");
  TORCH_CHECK(tile==8,"Training WMMA requires tile8");
  TORCH_CHECK(precision=="float16" || precision=="bfloat16","Training WMMA precision must be float16 or bfloat16");
  TORCH_CHECK(width>0 && height>0 && width<=2147483647LL && height<=2147483647LL &&
              width<=2147483647LL/height,"Training WMMA image dimensions exceed int32 pixel indexing");
  TORCH_CHECK(std::isfinite(threshold) && (float)threshold>0.f && (float)threshold<1.f,
              "Training WMMA threshold must remain in (0,1) after FP32 conversion");
  TORCH_CHECK(std::isfinite(beta) && std::isfinite((float)beta) &&
              std::isfinite(temperature) && std::isfinite((float)temperature),
              "Training WMMA gating parameters must be finite FP32 values");
  for(int i=0;i<17;++i) {
    TORCH_CHECK(x[i].is_cuda() && x[i].is_contiguous() && x[i].device()==x[0].device(),
                "Training WMMA inputs must be contiguous CUDA tensors on one device");
    TORCH_CHECK(x[i].scalar_type()==((i==5 || i==6)?at::kInt:at::kFloat),
                "Training WMMA requires FP32 inputs and int32 tile lists");
  }
  c10::cuda::CUDAGuard guard(x[0].device());
  const auto* props=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major>=8,"Training WMMA requires SM80 or newer");
  TORCH_CHECK(1+(width-1)/8<=props->maxGridSize[0] && 1+(height-1)/8<=props->maxGridSize[1],
              "Training WMMA tile grid exceeds device launch limits");
  const int64_t pixels=width*height;
  TORCH_CHECK(x[0].dim()==2 && x[0].size(0)==pixels && x[0].size(1)==3 && x[1].sizes()==x[0].sizes());
  TORCH_CHECK(x[2].dim()==2 && x[2].size(1)==3 && x[3].sizes()==x[2].sizes());
  const int64_t n=x[2].size(0);
  TORCH_CHECK(x[4].dim()==2 && x[4].size(0)==n && x[4].size(1)==4);
  TORCH_CHECK(x[7].dim()==1 && x[7].numel()==6 && x[8].sizes()==x[7].sizes());
  TORCH_CHECK(x[9].dim()==3 && x[9].size(0)==n && x[9].size(1)==32 && x[9].size(2)==72);
  TORCH_CHECK(x[10].dim()==2 && x[10].size(0)==n && x[10].size(1)==32);
  TORCH_CHECK(x[11].dim()==3 && x[11].size(0)==n && x[11].size(1)==32 && x[11].size(2)==32);
  TORCH_CHECK(x[12].sizes()==x[10].sizes());
  TORCH_CHECK(x[13].dim()==3 && x[13].size(0)==n && x[13].size(1)==4 && x[13].size(2)==32);
  TORCH_CHECK(x[14].dim()==2 && x[14].size(0)==n && x[14].size(1)==4);
  TORCH_CHECK(x[15].sizes()==x[0].sizes() && x[16].dim()==1 && x[16].numel()==pixels);
  validate_tile_candidate_capacity(x[5],x[6],width,height,8,8,MLPSP_MAX_TILE_PRIMS,
                                    "render_tiles_training_records_wmma",n);
  TORCH_CHECK(training_wmma_all_finite(x),"Training WMMA input is nonfinite; refusing unsafe forward/fallback");

  auto counts=torch::zeros({pixels},x[5].options());
  auto status=torch::zeros({},x[5].options());
  auto rgb=torch::zeros({pixels,3},x[0].options());
  auto alpha=torch::zeros({pixels,1},x[0].options());
  torch::Tensor offsets,ids,x6,raw,masks,gates;
  const dim3 grid(1+(width-1)/8,1+(height-1)/8,1);
  auto fallback=[&]() {
    training_wmma_frame_fallbacks.fetch_add(1);
    TORCH_WARN("Training WMMA numeric overflow: discarding both mixed passes and rerunning the whole FP32 record forward");
    // Status readback has completed the mixed launch. Release its frame-owned
    // storage before allocating the independent scalar count/emit result.
    offsets=torch::Tensor();ids=torch::Tensor();x6=torch::Tensor();raw=torch::Tensor();
    masks=torch::Tensor();gates=torch::Tensor();rgb=torch::Tensor();alpha=torch::Tensor();
    auto records=render_tiles_training_records_wrapper(x,width,height,tile,threshold,beta,temperature);
    if(!training_wmma_all_finite(records)) {
      training_wmma_fp32_nonfinite_failures.fetch_add(1);
      TORCH_CHECK(false,"Training WMMA scalar FP32 fallback records are nonfinite");
    }
    return records;
  };
  auto launch=[&](bool emit,int64_t total) {
#define TRAINING_WMMA_ARGS x[0].data_ptr<float>(),x[1].data_ptr<float>(),x[2].data_ptr<float>(),x[3].data_ptr<float>(),x[4].data_ptr<float>(),x[5].data_ptr<int32_t>(),x[6].data_ptr<int32_t>(),x[7].data_ptr<float>(),x[8].data_ptr<float>(),x[9].data_ptr<float>(),x[10].data_ptr<float>(),x[11].data_ptr<float>(),x[12].data_ptr<float>(),x[13].data_ptr<float>(),x[14].data_ptr<float>(),(int)n,(int)width,(int)height,(float)threshold,(float)beta,(float)temperature,emit,counts.data_ptr<int32_t>(),emit?offsets.data_ptr<int64_t>():nullptr,total,emit?ids.data_ptr<int32_t>():nullptr,emit?x6.data_ptr<float>():nullptr,emit?raw.data_ptr<float>():nullptr,emit?masks.data_ptr<int32_t>():nullptr,emit?gates.data_ptr<float>():nullptr,rgb.data_ptr<float>(),alpha.data_ptr<float>(),status.data_ptr<int>()
    if(precision=="bfloat16")
      render_training_wmma_kernel<__nv_bfloat16><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(TRAINING_WMMA_ARGS);
    else render_training_wmma_kernel<__half><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(TRAINING_WMMA_ARGS);
#undef TRAINING_WMMA_ARGS
    AT_CUDA_CHECK(cudaGetLastError());
    const int value=status.item<int>();
    training_wmma_check_structural_status(value);
    return value;
  };
  training_wmma_calls.fetch_add(1);
  if(launch(false,0)&TRAIN_WMMA_NUMERIC) return fallback();
  auto inclusive=torch::cumsum(counts,0,at::kLong);
  const int64_t total=inclusive[-1].item<int64_t>();
  TORCH_CHECK(total>=0 && total<=pixels*(int64_t)MLPSP_TILECENTER_MAX_HITS,
              "Training WMMA count total is outside the bounded allocation contract");
  offsets=torch::cat({torch::zeros({1},inclusive.options()),inclusive},0);
  ids=torch::empty({total},x[5].options());
  x6=torch::empty({total,6},x[0].options());
  raw=torch::empty({total,8},x[0].options());
  masks=torch::empty({total,2},x[5].options());
  gates=torch::empty({total,4},x[0].options());
  // Run even for total==0: emit validates every zero count and writes zero RGB/alpha.
  if(launch(true,total)&TRAIN_WMMA_NUMERIC) return fallback();
  return {rgb,alpha,ids,x6,raw,masks,gates,offsets};
}

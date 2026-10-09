// Tile-cooperative RGB inference for the 72->32->32->4 MLP.
// Four warps evaluate the same primitive for 64 tile rays with Tensor Cores.
// FP32 geometry, compositing order, and per-pixel termination are unchanged.
#include <mma.h>
#include <c10/cuda/CUDAGuard.h>
#include <atomic>

static std::atomic<int64_t> rgb_wmma_calls{0},rgb_wmma_fallbacks{0};
std::vector<int64_t> rgb_wmma_safety_stats() {
  return {rgb_wmma_calls.load(),rgb_wmma_fallbacks.load()};
}

template<typename T> __device__ __forceinline__ T rgb_mma_cast(float x);
template<> __device__ __forceinline__ __half rgb_mma_cast<__half>(float x) {
  return __float2half_rn(x);
}
template<> __device__ __forceinline__ __nv_bfloat16 rgb_mma_cast<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}
__device__ __forceinline__ float rgb_mma_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float rgb_mma_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T,int K,int COLS>
__device__ __forceinline__ void rgb_mma_layer(
    const T* a,const T* b,float* out,int warp) {
  using namespace nvcuda;
  for(int n=0;n<COLS;n+=16) {
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
#pragma unroll
    for(int k=0;k<K;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf;
      wmma::load_matrix_sync(af,a+warp*16*K+k,K);
      wmma::load_matrix_sync(bf,b+n*K+k,K);
      wmma::mma_sync(acc,af,bf,acc);
    }
    wmma::store_matrix_sync(out+warp*16*32+n,acc,32,wmma::mem_row_major);
  }
}

template<typename T,int K>
__device__ __forceinline__ void rgb_mma_first_layer_alias(
    const T* a,const T* b,float* out,int warp) {
  using namespace nvcuda;
  // Keep both output-column tiles live until every warp has consumed A/W1.
  // Unlike the later layers, out aliases A: storing the first 16 columns
  // early would corrupt rows that this or another warp has not yet loaded.
  wmma::fragment<wmma::accumulator,16,16,16,float> left,right;
  wmma::fill_fragment(left,0.f);
  wmma::fill_fragment(right,0.f);
#pragma unroll
  for(int k=0;k<K;k+=16) {
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf;
    wmma::load_matrix_sync(af,a+warp*16*K+k,K);
    wmma::load_matrix_sync(bf,b+k,K);
    wmma::mma_sync(left,af,bf,left);
  }
#pragma unroll
  for(int k=0;k<K;k+=16) {
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> bf;
    wmma::load_matrix_sync(af,a+warp*16*K+k,K);
    wmma::load_matrix_sync(bf,b+16*K+k,K);
    wmma::mma_sync(right,af,bf,right);
  }
  // All 128 threads call this helper on the same block-uniform hit branch.
  // This CTA barrier must precede ANY store through the aliased out pointer.
  __syncthreads();
  wmma::store_matrix_sync(out+warp*16*32,left,32,wmma::mem_row_major);
  wmma::store_matrix_sync(out+warp*16*32+16,right,32,wmma::mem_row_major);
}

template<typename T>
__global__ __launch_bounds__(128,2) void render_rgb_wmma_kernel(
    const float* rays_o,const float* rays_d,const float* pos,const float* logs,const float* quats,
    const int32_t* candidates,const int32_t* tile_offsets,const float* freq,const float* att,
    const float* W1,const float* b1,const float* W2,const float* b2,const float* W3,const float* b3,
    int N,int width,int height,float threshold,float* rgb,float* out_alpha,int* nonfinite) {
  constexpr int threads=128,capacity=MLPSP_MAX_TILE_PRIMS,items=capacity/threads;
  static_assert(capacity%threads==0,"WMMA tile sort requires a multiple of 128 candidates");
  using Sort=cub::BlockRadixSort<float,threads,items,int>;
  __shared__ int prims[capacity];
  // Non-overlapping lifetimes: sort -> input+W1 -> activation+hidden. All
  // aliases change ownership only after a full block barrier. First-layer
  // output replaces the exhausted input, saving another 8 KiB/CTA.
  static_assert(sizeof(typename Sort::TempStorage)<=sizeof(T)*(64*80+32*80),"sort scratch size");
  static_assert(sizeof(float)*64*32<=sizeof(T)*64*80,"activation must fit exhausted input storage");
  __shared__ __align__(32) T staging[64*80+32*80];
  auto& sort_temp=*reinterpret_cast<typename Sort::TempStorage*>(staging);
  T* input=staging;
  T* weight1=staging+64*80;
  T* hidden=weight1;
  __shared__ __align__(32) T weight2[32*32];
  __shared__ __align__(32) T weight3[16*32];
  float* activation=reinterpret_cast<float*>(input);
  __shared__ float bias1[32],bias2[32],bias3[4];
  __shared__ bool active[64];
  bool bad=false;
  int tid=threadIdx.x,warp=tid/32;
  int tile=blockIdx.y*((width+7)/8)+blockIdx.x;
  int x=blockIdx.x*8+tid%8,y=blockIdx.y*8+tid/8;
  bool pixel=tid<64 && x<width && y<height;
  int pix=pixel?y*width+x:0;
  int center=min(int(blockIdx.y)*8+4,height-1)*width+min(int(blockIdx.x)*8+4,width-1);
  float ox=pixel?rays_o[pix*3]:0.f,oy=pixel?rays_o[pix*3+1]:0.f,oz=pixel?rays_o[pix*3+2]:0.f;
  float rx=pixel?rays_d[pix*3]:0.f,ry=pixel?rays_d[pix*3+1]:0.f,rz=pixel?rays_d[pix*3+2]:0.f;
  int begin=tile_offsets[tile],end=tile_offsets[tile+1],length=end-begin;
  float keys[items];
  int values[items];
#pragma unroll
  for(int item=0;item<items;++item) {
    int j=tid*items+item;
    if(j<length) {
      int p=candidates[begin+j];
      float t=tstar_key_tilecenter(rays_o[center*3],rays_o[center*3+1],rays_o[center*3+2],
        rays_d[center*3],rays_d[center*3+1],rays_d[center*3+2],pos[p*3],pos[p*3+1],pos[p*3+2],
        logs[p*3],logs[p*3+1],logs[p*3+2],quats[p*4],quats[p*4+1],quats[p*4+2],quats[p*4+3]);
      keys[item]=t>0.f?t:1.e30f;
      values[item]=p;
    } else { keys[item]=1.e30f;values[item]=-1; }
  }
  Sort(sort_temp).Sort(keys,values);
  __syncthreads();
#pragma unroll
  for(int item=0;item<items;++item) prims[tid*items+item]=values[item];
  __syncthreads();
  float3 color=make_float3(0.f,0.f,0.f);
  float trans=1.f;
  for(int j=0;j<length;++j) {
    int p=prims[j];
    if(p<0 || p>=N) continue; // Block-uniform branch.
    float t,dx=0.f,dy=0.f,dz=0.f,qlx,qly,qlz,decay=0.f;
    bool hit=false;
    if(pixel && trans>=1.e-3f) {
      hit=hit_and_features_rot(ox,oy,oz,rx,ry,rz,pos[p*3],pos[p*3+1],pos[p*3+2],
        logs[p*3],logs[p*3+1],logs[p*3+2],quats[p*4],quats[p*4+1],quats[p*4+2],quats[p*4+3],
        threshold,t,dx,dy,dz,qlx,qly,qlz,decay);
    }
    if(!__syncthreads_or(hit)) continue;
    // One ray owner computes the unchanged FP32 positional encoding. Inactive
    // rays participate in MMA with zeros and never modify their accumulator.
    if(tid<64) {
      active[tid]=hit;
      float v[6]={rx,ry,rz,dx,dy,dz};
#pragma unroll
      for(int c=0;c<6;++c) {
#pragma unroll
        for(int f=0;f<6;++f) {
          float phase=v[c]*freq[f]*att[f];
          bad |= hit && !isfinite(phase);
          input[tid*80+c*6+f]=rgb_mma_cast<T>(hit?__sinf(phase):0.f);
          input[tid*80+36+c*6+f]=rgb_mma_cast<T>(hit?__cosf(phase):0.f);
        }
      }
#pragma unroll
      for(int k=72;k<80;++k) input[tid*80+k]=rgb_mma_cast<T>(0.f);
    }
    // Weights are shared across all 64 rays. Padding applies only to GEMM
    // operands; neither the learned dimensions nor master weights are changed.
    for(int q=tid;q<32*80;q+=threads) {
      int h=q/80,k=q%80;
      weight1[q]=rgb_mma_cast<T>(k<72?W1[p*2304+h*72+k]:0.f);
      bad |= !isfinite(rgb_mma_float(weight1[q]));
    }
    for(int q=tid;q<32*32;q+=threads) {
      weight2[q]=rgb_mma_cast<T>(W2[p*1024+q]);
      bad |= !isfinite(rgb_mma_float(weight2[q]));
    }
    for(int q=tid;q<16*32;q+=threads) {
      weight3[q]=rgb_mma_cast<T>(q<4*32?W3[p*128+q]:0.f);
      bad |= !isfinite(rgb_mma_float(weight3[q]));
    }
    if(tid<32) { bias1[tid]=b1[p*32+tid];bias2[tid]=b2[p*32+tid]; }
    if(tid<4) bias3[tid]=b3[p*4+tid];
    __syncthreads();
    rgb_mma_first_layer_alias<T,80>(input,weight1,activation,warp);
    // W1 is shared by every warp and aliases hidden. No warp may begin its
    // hidden writes until every warp has finished reading the complete W1.
    __syncthreads();
    // Each WMMA warp owns all 32 channels of its 16 rows. The intermediate
    // activation/hidden reads and writes below therefore stay within one warp;
    // other warps may advance independently without cross-row dependencies.
    for(int q=warp*16*32+(tid&31);q<(warp+1)*16*32;q+=32) {
      float value=activation[q]+bias1[q%32];
      hidden[q]=rgb_mma_cast<T>(fmaxf(value,0.f));
      bad |= active[q/32] && (!isfinite(value) || !isfinite(rgb_mma_float(hidden[q])));
    }
    __syncwarp(0xffffffffu);
    rgb_mma_layer<T,32,32>(hidden,weight2,activation,warp);
    __syncwarp(0xffffffffu);
    for(int q=warp*16*32+(tid&31);q<(warp+1)*16*32;q+=32) {
      float value=activation[q]+bias2[q%32];
      hidden[q]=rgb_mma_cast<T>(fmaxf(value,0.f));
      bad |= active[q/32] && (!isfinite(value) || !isfinite(rgb_mma_float(hidden[q])));
    }
    __syncwarp(0xffffffffu);
    rgb_mma_layer<T,32,16>(hidden,weight3,activation,warp);
    // RGB/alpha are consumed by the original one-thread-per-pixel owners,
    // which are not the same warps that produced all of their WMMA rows.
    __syncthreads();
    if(hit) {
#pragma unroll
      for(int c=0;c<4;++c) bad |= !isfinite(activation[tid*32+c]+bias3[c]);
      float R=fast_sigmoid(activation[tid*32]+bias3[0]);
      float G=fast_sigmoid(activation[tid*32+1]+bias3[1]);
      float B=fast_sigmoid(activation[tid*32+2]+bias3[2]);
      float sigma_raw=fast_softplus(activation[tid*32+3]+bias3[3]);
      float alpha=1.f-__expf(-sigma_raw*decay);
      float w=trans*alpha;
      color.x=fmaf(w,R,color.x);color.y=fmaf(w,G,color.y);color.z=fmaf(w,B,color.z);
      trans*=1.f-alpha;
    }
    // Uniform termination/barrier: no warp may exit while another uses WMMA.
    if(!__syncthreads_or(pixel && trans>=1.e-3f)) break;
  }
  if(bad) atomicExch(nonfinite,1);
  if(pixel) {
    rgb[pix*3]=color.x;rgb[pix*3+1]=color.y;rgb[pix*3+2]=color.z;
    out_alpha[pix]=1.f-trans;
  }
}

std::vector<torch::Tensor> render_tiles_rgb_wmma_wrapper(
    const std::vector<torch::Tensor>& x,int64_t width,int64_t height,double threshold,bool bf16) {
  TORCH_CHECK(x.size()==15,"WMMA requires the 15 scene tensors");
  TORCH_CHECK(width>0 && height>0 && threshold>0. && threshold<1.);
  TORCH_CHECK(width<=2147483647LL && height<=2147483647LL &&
              width<=2147483647LL/height,"WMMA image dimensions exceed int32 pixel indexing");
  for(int i=0;i<15;++i) {
    TORCH_CHECK(x[i].is_cuda() && x[i].is_contiguous() && x[i].device()==x[0].device());
    TORCH_CHECK(x[i].scalar_type()==((i==5 || i==6)?at::kInt:at::kFloat));
  }
  c10::cuda::CUDAGuard guard(x[0].device());
  const auto* props=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major>=8,"WMMA rendering requires SM80 or newer");
  int64_t pixels=width*height,n=x[2].size(0);
  TORCH_CHECK(x[0].dim()==2 && x[0].size(0)==pixels && x[0].size(1)==3 && x[1].sizes()==x[0].sizes());
  TORCH_CHECK(x[2].dim()==2 && x[2].size(1)==3 && x[3].sizes()==x[2].sizes());
  TORCH_CHECK(x[4].dim()==2 && x[4].size(0)==n && x[4].size(1)==4);
  TORCH_CHECK(x[7].dim()==1 && x[7].numel()==6 && x[8].sizes()==x[7].sizes());
  TORCH_CHECK(x[9].dim()==3 && x[9].size(0)==n && x[9].size(1)==32 && x[9].size(2)==72);
  TORCH_CHECK(x[10].dim()==2 && x[10].size(0)==n && x[10].size(1)==32);
  TORCH_CHECK(x[11].dim()==3 && x[11].size(0)==n && x[11].size(1)==32 && x[11].size(2)==32);
  TORCH_CHECK(x[12].sizes()==x[10].sizes());
  TORCH_CHECK(x[13].dim()==3 && x[13].size(0)==n && x[13].size(1)==4 && x[13].size(2)==32);
  TORCH_CHECK(x[14].dim()==2 && x[14].size(0)==n && x[14].size(1)==4);
  validate_tile_candidate_capacity(x[5],x[6],width,height,8,8,MLPSP_MAX_TILE_PRIMS,"render_tiles_rgb_wmma",n);
  auto rgb=torch::empty({pixels,3},x[0].options());
  auto alpha=torch::empty({pixels,1},x[0].options());
  auto nonfinite=torch::zeros({},x[0].options().dtype(at::kInt));
  rgb_wmma_calls.fetch_add(1);
  dim3 grid((width+7)/8,(height+7)/8,1);
#define RGB_WMMA_ARGS x[0].data_ptr<float>(),x[1].data_ptr<float>(),x[2].data_ptr<float>(),x[3].data_ptr<float>(),x[4].data_ptr<float>(),x[5].data_ptr<int32_t>(),x[6].data_ptr<int32_t>(),x[7].data_ptr<float>(),x[8].data_ptr<float>(),x[9].data_ptr<float>(),x[10].data_ptr<float>(),x[11].data_ptr<float>(),x[12].data_ptr<float>(),x[13].data_ptr<float>(),x[14].data_ptr<float>(),(int)n,(int)width,(int)height,(float)threshold,rgb.data_ptr<float>(),alpha.data_ptr<float>()
  if(bf16) render_rgb_wmma_kernel<__nv_bfloat16><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(RGB_WMMA_ARGS,nonfinite.data_ptr<int>());
  else render_rgb_wmma_kernel<__half><<<grid,128,0,at::cuda::getCurrentCUDAStream()>>>(RGB_WMMA_ARGS,nonfinite.data_ptr<int>());
#undef RGB_WMMA_ARGS
  AT_CUDA_CHECK(cudaGetLastError());
  if(nonfinite.item<int>()) {
    // Catch before ReLU's fmaxf can hide NaNs. Never accept a finite-looking
    // image obtained by silently zeroing an overflowing hidden activation.
    for(int i=0;i<15;++i) if(i!=5 && i!=6)
      TORCH_CHECK(torch::isfinite(x[i]).all().item<bool>(),"WMMA input is nonfinite; refusing fallback");
    TORCH_WARN_ONCE("WMMA operand/activation overflow: rendering this frame with scalar FP32 instead");
    rgb_wmma_fallbacks.fetch_add(1);
    auto fallback=render_tiles_forward_tilecenter_wrapper(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],
      x[9],x[10],x[11],x[12],x[13],x[14],width,height,8,8,threshold);
    TORCH_CHECK(torch::isfinite(std::get<0>(fallback)).all().item<bool>() &&
                torch::isfinite(std::get<1>(fallback)).all().item<bool>(),"Scalar FP32 fallback output is nonfinite");
    return {std::get<0>(fallback),std::get<1>(fallback)};
  }
  return {rgb,alpha};
}

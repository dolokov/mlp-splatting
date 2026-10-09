// Reverse compositing over saved FP32 forward records.
// Included after the renderer helpers; no MLP recomputation and no weight atomics.
__global__ void primitive_records_reverse_kernel(
    const int32_t* ids,const int64_t* offsets,const float* x6,const float* raw,
    const float* logs,const float* quats,const float* grgb,const float* gt,
    int pixels,float* d4,float* gp,float* gs,float* gq) {
  int pix=blockIdx.x*blockDim.x+threadIdx.x;
  if(pix>=pixels) return;
  float next=gt[pix];
  for(int64_t r=offsets[pix+1]-1;r>=offsets[pix];--r) {
    int p=ids[r];
    const float* v=raw+r*8;
    float R=v[0],G=v[1],B=v[2],sigma_raw=v[3],trans=v[4],alpha=v[5],decay=v[6];
    float dotg=dot3(grgb[pix*3],grgb[pix*3+1],grgb[pix*3+2],R,G,B);
    float grad_alpha=trans*(dotg-next);
    next=next*(1.f-alpha)+dotg*alpha;
    float grad_sigma=grad_alpha*__expf(-sigma_raw*decay);
    float grad_decay=grad_sigma*sigma_raw;
    d4[r*4]=grgb[pix*3]*(trans*alpha)*R*(1.f-R);
    d4[r*4+1]=grgb[pix*3+1]*(trans*alpha)*G*(1.f-G);
    d4[r*4+2]=grgb[pix*3+2]*(trans*alpha)*B*(1.f-B);
    d4[r*4+3]=grad_sigma*decay*fast_sigmoid(v[7]);
    float dx=x6[r*6+3],dy=x6[r*6+4],dz=x6[r*6+5];
    float qw=quats[p*4],qx=quats[p*4+1],qy=quats[p*4+2],qz=quats[p*4+3];
    float3 q=quat_rotate_conj(qw,qx,qy,qz,make_float3(dx,dy,dz));
    float mx=__expf(-2.f*logs[p*3]),my=__expf(-2.f*logs[p*3+1]),mz=__expf(-2.f*logs[p*3+2]);
    float gx=-grad_decay*decay*mx*q.x;
    float gy=-grad_decay*decay*my*q.y;
    float gz=-grad_decay*decay*mz*q.z;
    float3 gd=quat_rotate(qw,qx,qy,qz,make_float3(gx,gy,gz));
    atomic_add_warp_agg(gp+p*3,gd.x);
    atomic_add_warp_agg(gp+p*3+1,gd.y);
    atomic_add_warp_agg(gp+p*3+2,gd.z);
    atomic_add_warp_agg(gs+p*3,grad_decay*decay*mx*q.x*q.x);
    atomic_add_warp_agg(gs+p*3+1,grad_decay*decay*my*q.y*q.y);
    atomic_add_warp_agg(gs+p*3+2,grad_decay*decay*mz*q.z*q.z);
    float dqw,dqx,dqy,dqz;
    rotmat_grad_to_quat_grad(qw,qx,qy,qz,
      dx*gx,dx*gy,dx*gz,dy*gx,dy*gy,dy*gz,dz*gx,dz*gy,dz*gz,
      dqw,dqx,dqy,dqz);
    atomic_add_warp_agg(gq+p*4,dqw);
    atomic_add_warp_agg(gq+p*4+1,dqx);
    atomic_add_warp_agg(gq+p*4+2,dqy);
    atomic_add_warp_agg(gq+p*4+3,dqz);
  }
}

std::vector<torch::Tensor> primitive_records_reverse_wrapper(
    torch::Tensor ids,torch::Tensor offsets,torch::Tensor x6,torch::Tensor raw,
    torch::Tensor logs,torch::Tensor quats,torch::Tensor grgb,torch::Tensor gt) {
  for(const auto& t: {ids,offsets,x6,raw,logs,quats,grgb,gt}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==ids.device());
  }
  TORCH_CHECK(ids.scalar_type()==at::kInt && offsets.scalar_type()==at::kLong);
  for(const auto& t: {x6,raw,logs,quats,grgb,gt}) TORCH_CHECK(t.scalar_type()==at::kFloat);
  int64_t total=ids.numel(),pixels=grgb.size(0),n=logs.size(0);
  TORCH_CHECK(ids.dim()==1 && offsets.dim()==1 && offsets.numel()==pixels+1);
  TORCH_CHECK(x6.dim()==2 && x6.size(0)==total && x6.size(1)==6);
  TORCH_CHECK(raw.dim()==2 && raw.size(0)==total && raw.size(1)==8);
  TORCH_CHECK(logs.dim()==2 && logs.size(1)==3);
  TORCH_CHECK(quats.dim()==2 && quats.size(0)==n && quats.size(1)==4);
  TORCH_CHECK(grgb.dim()==2 && grgb.size(1)==3 && gt.dim()==1 && gt.numel()==pixels);
  TORCH_CHECK(offsets[0].item<int64_t>()==0 && offsets[-1].item<int64_t>()==total);
  TORCH_CHECK(torch::all(offsets.slice(0,1)>=offsets.slice(0,0,-1)).item<bool>());
  if(total) TORCH_CHECK(ids.min().item<int32_t>()>=0 && ids.max().item<int32_t>()<n);
  auto d4=torch::empty({total,4},x6.options());
  auto gp=torch::zeros_like(logs),gs=torch::zeros_like(logs),gq=torch::zeros_like(quats);
  if(pixels) primitive_records_reverse_kernel<<<(pixels+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
    ids.data_ptr<int32_t>(),offsets.data_ptr<int64_t>(),x6.data_ptr<float>(),raw.data_ptr<float>(),
    logs.data_ptr<float>(),quats.data_ptr<float>(),grgb.data_ptr<float>(),gt.data_ptr<float>(),
    (int)pixels,d4.data_ptr<float>(),gp.data_ptr<float>(),gs.data_ptr<float>(),gq.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
  return {d4,gp,gs,gq};
}

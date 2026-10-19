// Reference-oriented native MLP rendering. Brute-force candidates; overflow is explicit.
@group(0) @binding(0) var<storage, read> scene: array<f32>;
@group(0) @binding(1) var<storage, read> camera: array<f32>;
@group(0) @binding(2) var image: texture_storage_2d<rgba8unorm, write>;
const N: u32 = $N;
const H: u32 = $H;
const F: u32 = $F;
const INPUT: u32 = F * 12u;
const STRIDE: u32 = 10u + H*INPUT + H + H*H + H + 4u*H + 4u;
fn rotate_inverse(q: vec4f, v: vec3f) -> vec3f {
  let u = -q.yzw;
  return v + 2.0 * cross(u, cross(u, v) + q.x * v);
}
fn v3(i: u32) -> vec3f { return vec3f(scene[i], scene[i+1u], scene[i+2u]); }
fn evaluate(p: u32, direction: vec3f, offset: vec3f) -> vec4f {
  let base = p*STRIDE + 10u;
  var x: array<f32, $INPUT>;
  var values = array<f32, 6>(direction.x,direction.y,direction.z,offset.x,offset.y,offset.z);
  for(var c=0u;c<6u;c++) { for(var f=0u;f<F;f++) {
    let phase=values[c]*scene[N*STRIDE+f]*scene[N*STRIDE+F+f];
    x[c*F+f]=sin(phase); x[6u*F+c*F+f]=cos(phase);
  }}
  var h1: array<f32, $H>;
  var h2: array<f32, $H>;
  for(var h=0u;h<H;h++) {
    var s=scene[base+H*INPUT+h];
    for(var j=0u;j<INPUT;j++) { s+=scene[base+h*INPUT+j]*x[j]; }
    h1[h]=max(0.0,s);
  }
  let b2=base+H*INPUT+H;
  for(var h=0u;h<H;h++) {
    var s=scene[b2+H*H+h];
    for(var j=0u;j<H;j++) { s+=scene[b2+h*H+j]*h1[j]; }
    h2[h]=max(0.0,s);
  }
  let b3=b2+H*H+H;
  var result: vec4f;
  for(var o=0u;o<4u;o++) {
    var s=scene[b3+4u*H+o];
    for(var j=0u;j<H;j++) { s+=scene[b3+o*H+j]*h2[j]; }
    if(o<3u) { result[o]=1.0/(1.0+exp(-s)); }
    else { result[o]=max(s,0.0)+log(1.0+exp(-abs(s))); }
  }
  return result;
}
@compute @workgroup_size(4,4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims=textureDimensions(image);
  if(id.x>=dims.x || id.y>=dims.y) { return; }
  let right=vec3f(camera[0],camera[1],camera[2]);
  let down=vec3f(camera[4],camera[5],camera[6]);
  let forward=vec3f(camera[8],camera[9],camera[10]);
  let origin=vec3f(camera[12],camera[13],camera[14]);
  let xy=(vec2f(id.xy)+0.5)/vec2f(dims);
  let direction=normalize(right*((xy.x-camera[18])/camera[16])+down*((xy.y-camera[19])/camera[17])+forward);
  var depths: array<f32,128>;
  var ids: array<u32,128>;
  var offsets: array<vec3f,128>;
  var decays: array<f32,128>;
  var count=0u;
  for(var p=0u;p<N;p++) {
    let b=p*STRIDE;
    let center=v3(b); let precision=exp(-2.0*v3(b+3u));
    let q=vec4f(scene[b+6u],scene[b+7u],scene[b+8u],scene[b+9u]);
    let local_v=rotate_inverse(q,center-origin);
    let local_d=rotate_inverse(q,direction);
    let denom=dot(local_d*local_d,precision);
    if(denom<=1e-20) { continue; }
    let t=dot(local_d*local_v,precision)/denom;
    if(t<=0.0) { continue; }
    let offset=center-origin-t*direction;
    let local=rotate_inverse(q,offset);
    let decay=exp(-0.5*max(dot(local*local,precision),0.0));
    if(decay<=camera[20]) { continue; }
    if(count==128u) { textureStore(image,vec2i(id.xy),vec4f(1,0,1,1)); return; }
    var j=count;
    loop {
      if(j==0u) { break; }
      if(depths[j-1u]<=t) { break; }
      depths[j]=depths[j-1u]; ids[j]=ids[j-1u]; offsets[j]=offsets[j-1u]; decays[j]=decays[j-1u]; j--;
    }
    depths[j]=t; ids[j]=p; offsets[j]=offset; decays[j]=decay; count++;
  }
  var trans=1.0; var rgb=vec3f(0);
  for(var j=0u;j<count;j++) {
    let value=evaluate(ids[j],direction,offsets[j]);
    let alpha=1.0-exp(-value.w*decays[j]);
    rgb+=trans*alpha*value.xyz; trans*=1.0-alpha;
    if(trans<0.001) { break; }
  }
  textureStore(image,vec2i(id.xy),vec4f(rgb,1));
}

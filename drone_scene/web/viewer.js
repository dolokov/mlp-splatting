const status=document.querySelector('#status'), canvas=document.querySelector('canvas');
const $=s=>document.querySelector(s);
const fail=e=>{status.textContent=`Error: ${e.message || e}`; console.error(e);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const n=Math.hypot(...a);return a.map(x=>x/n);};
const rotate=(v,axis,t)=>{const c=Math.cos(t),s=Math.sin(t),d=v.reduce((a,x,i)=>a+x*axis[i],0),k=cross(axis,v);return v.map((x,i)=>x*c+k[i]*s+axis[i]*d*(1-c));};
async function fetchOK(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r;}
async function main(){
 if(!navigator.gpu)throw Error('WebGPU unavailable. Use a WebGPU-capable browser on localhost/HTTPS. No server-rendering fallback.');
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
 if(!adapter)throw Error('No WebGPU adapter available on this browser/device.');
 const manifest=await (await fetchOK('scene.json')).json();
 if(manifest.version!==1)throw Error('Unsupported scene export version');
 const bytes=await (await fetchOK('scene.bin')).arrayBuffer();
 if(bytes.byteLength!==manifest.byte_length)throw Error('Scene binary length mismatch');
 if(bytes.byteLength>adapter.limits.maxStorageBufferBindingSize || bytes.byteLength>adapter.limits.maxBufferSize)throw Error('Scene exceeds browser GPU buffer limits');
 const device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.max(134217728,bytes.byteLength),maxBufferSize:Math.max(268435456,bytes.byteLength)}});
 device.lost.then(info=>fail(Error(`GPU device lost: ${info.message}`)));
 device.addEventListener('uncapturederror',e=>fail(e.error));
 const scene=device.createBuffer({size:bytes.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(scene,0,bytes);
 const camera=device.createBuffer({size:96,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 let code=await (await fetchOK('render.wgsl')).text();
 const replacements={N:manifest.count,H:manifest.hidden,F:manifest.frequencies,INPUT:manifest.frequencies*12};
 code=code.replace(/\$(INPUT|N|H|F)\b/g,(_,key)=>String(replacements[key]));
 const module=device.createShaderModule({code});
 const info=await module.getCompilationInfo();const errors=info.messages.filter(x=>x.type==='error');
 if(errors.length)throw Error(errors.map(x=>`${x.lineNum}:${x.linePos}: ${x.message}`).join('\n'));
 const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
 // GPU presentation keeps rendered pixels on the GPU; no image streaming/readback per frame.
 const display=device.createShaderModule({code:`@group(0) @binding(0) var img:texture_2d<f32>;
 @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
 @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return textureLoad(img,vec2i(p.xy),0);}`});
 const context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const displayPipeline=device.createRenderPipeline({layout:'auto',vertex:{module:display,entryPoint:'vs'},fragment:{module:display,entryPoint:'fs',targets:[{format}]}});
 let pose,texture,busy=false,pending=false;
 manifest.cameras.forEach((p,i)=>{$('#poses').add(new Option(p.name,i));});
 const reset=()=>{pose=manifest.cameras[Number($('#poses').value)].pose.map(r=>r.slice());request().catch(fail);};
 async function request(){
  if(busy){pending=true;return;}busy=true;
  try{
   const width=Number($('#resolution').value),height=Math.max(1,Math.round(width*manifest.camera.height/manifest.camera.width));
   if(canvas.width!==width||canvas.height!==height||!texture){canvas.width=width;canvas.height=height;texture?.destroy();texture=device.createTexture({size:[width,height],format:'rgba8unorm',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});}
   const uniforms=new Float32Array(24);
   for(let c=0;c<4;c++)for(let r=0;r<3;r++)uniforms[c*4+r]=pose[r][c];
   uniforms.set([manifest.camera.fx/manifest.camera.width,manifest.camera.fy/manifest.camera.height,manifest.camera.cx/manifest.camera.width,manifest.camera.cy/manifest.camera.height,manifest.threshold],16);
   device.queue.writeBuffer(camera,0,uniforms);
   const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:scene}},{binding:1,resource:{buffer:camera}},{binding:2,resource:texture.createView()}]});
   const start=performance.now(),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(width/4),Math.ceil(height/4));pass.end();
   const render=encoder.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
   render.setPipeline(displayPipeline);render.setBindGroup(0,device.createBindGroup({layout:displayPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:texture.createView()}]}));render.draw(3);render.end();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
   status.textContent=`${manifest.count} primitives · ${width}×${height} · ${(performance.now()-start).toFixed(0)} ms (GPU completion + submission)\nAdapter: ${adapter.info?.description || adapter.info?.vendor || 'available'}\nExperimental: no CUDA pixel-parity certification. Black background. Magenta indicates overflow.`;
  }finally{busy=false;if(pending){pending=false;request().catch(fail);}}
 }
 $('#render').onclick=()=>request().catch(fail);$('#reset').onclick=reset;$('#poses').onchange=reset;$('#resolution').onchange=()=>request().catch(fail);
 $('#save').onclick=()=>canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='mlp-webgpu.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 canvas.onkeydown=e=>{const move={w:[2,1],s:[2,-1],a:[0,-1],d:[0,1],q:[1,1],e:[1,-1]}[e.key.toLowerCase()];if(!move)return;e.preventDefault();const speed=Number($('#speed').value);if(!Number.isFinite(speed)||speed<=0)return;for(let r=0;r<3;r++)pose[r][3]+=pose[r][move[0]]*move[1]*speed;request().catch(fail);};
 let last;canvas.onpointerdown=e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);canvas.focus();};canvas.onpointerup=()=>last=null;
 canvas.onpointermove=e=>{if(!last)return;const dx=(e.clientX-last[0])*.003,dy=(e.clientY-last[1])*.003;last=[e.clientX,e.clientY];let cols=[0,1,2].map(c=>pose.map(r=>r[c]));cols=cols.map(v=>rotate(v,cols[1],dx));const axis=norm(cols[0]);cols=cols.map(v=>rotate(v,axis,-dy));for(let c=0;c<3;c++)for(let r=0;r<3;r++)pose[r][c]=cols[c][r];request().catch(fail);};
 reset();
}
main().catch(fail);

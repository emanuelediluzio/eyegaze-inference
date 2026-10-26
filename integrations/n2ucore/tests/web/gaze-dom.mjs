export class Element {
  constructor(tag) { this.tag=tag;this.children=[];this.style={};this.events={};this.attributes={};this.hidden=false;this.disabled=false;this.isConnected=true;this.classList={toggle(){}}; }
  append(...children) { for(const c of children){c.parentElement=this;this.children.push(c);} }
  replaceChildren(...children) { this.children=[];this.append(...children); }
  setAttribute(k,v){this.attributes[k]=v;}
  addEventListener(k,fn){this.events[k]=fn;}
  removeEventListener(k){delete this.events[k];}
  remove(){this.isConnected=false;if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(c=>c!==this);}
  replaceWith(other){const parent=this.parentElement;if(parent){const i=parent.children.indexOf(this);parent.children[i]=other;other.parentElement=parent;}}
  focus(){}
  click(){if(!this.disabled){this.events.click?.();this.onclick?.();}}
  transferControlToOffscreen(){return {};}
}
export function find(root,text){if(root.textContent===text)return root;for(const child of root.children??[]){const result=find(child,text);if(result)return result;}return null;}
export function descendants(root,tag){return [...(root.tag===tag?[root]:[]),...(root.children??[]).flatMap(c=>descendants(c,tag))];}
export function environment() {
 const old={document:globalThis.document,window:globalThis.window,requestAnimationFrame:globalThis.requestAnimationFrame,setTimeout:globalThis.setTimeout};
 function eventTarget(object) {
  const handlers=new Map();object.events={};
  object.addEventListener=(key,fn)=>{if(!handlers.has(key))handlers.set(key,new Set());handlers.get(key).add(fn);object.events[key]=(...args)=>{for(const listener of [...handlers.get(key)])listener(...args);};};
  object.removeEventListener=(key,fn)=>{handlers.get(key)?.delete(fn);if(!handlers.get(key)?.size){handlers.delete(key);delete object.events[key];}};
  return object;
 }
 globalThis.document=eventTarget({createElement:t=>new Element(t),createElementNS:(_ns,t)=>new Element(t),createTextNode:text=>Object.assign(new Element('text'),{textContent:text}),body:new Element('body'),head:new Element('head'),querySelector:()=>null,hidden:false});
 globalThis.window=eventTarget({innerWidth:1000,innerHeight:600,screenX:0,screenY:0,devicePixelRatio:1});
 globalThis.requestAnimationFrame=fn=>fn();
 globalThis.setTimeout=(fn,ms)=>old.setTimeout(fn,ms===3000?0:ms);
 const views=new Map();const calls=[];
 const renderer={show(id,channels,canvas,options,handlers){views.set(id,{channels,canvas,options,handlers});calls.push(['show',id]);return true;},hide(id){views.delete(id);calls.push(['hide',id]);},control(...args){calls.push(['control',...args]);}};
 return {views,calls,renderer,restore(){Object.assign(globalThis,old);}};
}
export async function until(predicate){for(let i=0;i<300;i++){if(predicate())return;await new Promise(r=>setTimeout(r,2));}throw new Error('UI did not settle');}
export const direction={schema:'gaze-direction.v1',status:'ok',yaw_rad:.2,pitch_rad:-.1,yaw_deg:11.46,pitch_deg:-5.73,bbox_xyxy_px:[100,80,300,320],image_size_px:[640,480],eyes_px:[[155,160],[245,160]]};
export const point={schema:'screen-point-of-gaze.v1',status:'ok',screen_xy_px:[400,250],screen_size_px:[1000,600]};
export const calibration={cx:[500,100,0,0,0,0],cy:[300,0,100,0,0,0],screen_w:1000,screen_h:600};

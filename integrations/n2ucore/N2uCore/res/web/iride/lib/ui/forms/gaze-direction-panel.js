import {el,gazeStyles,GazeStream,statusText} from "./gaze-ui.js";
let nextVideo=0;
export class GazeDirectionPanel {
  static channels=["GazeDirection"];
  static preferredWidth=780;
  #o; #root; #status; #canvas; #svg; #stats; #details; #stream; #timer; #last=0; #sample=null; #alive; #paused=false; #released=false; #videoId; #videoShown=false; #box=true; #arrows=true; #eyes=true; #rate=0; #metricValues=[]; #detailsBox;
  constructor(options) { this.#o=options; this.#alive=options.alive ?? true; this.#videoId="gaze-video-"+(++nextVideo); }
  render(container) {
    gazeStyles();this.#root=el("section","gaze-panel gaze-direction");
    const header=el("header","gaze-heading"); header.append(el("strong","","Direzione dello sguardo")); this.#status=el("span","gaze-status","In attesa del volto");header.append(this.#status);
    const layout=el("div","gaze-layout");const main=el("div","gaze-main");const stage=el("div","gaze-video-stage");
    this.#canvas=el("canvas","gaze-video"); this.#canvas.setAttribute("aria-label","Video della webcam e direzione dello sguardo");
    this.#svg=document.createElementNS("http://www.w3.org/2000/svg","svg");this.#svg.setAttribute("class","gaze-video-overlay");this.#svg.setAttribute("aria-hidden","true");
    stage.append(this.#canvas,this.#svg);main.append(stage);
    const controls=el("div","gaze-switches");
    for(const [label,value,set] of [["Riquadro volto",true,v=>this.#box=v],["Frecce",true,v=>this.#arrows=v],["Punti occhi",true,v=>this.#eyes=v],["Specchio",false,v=>stage.classList.toggle("mirrored",v)]]) {
      const row=el("label","field");const input=el("input");input.type="checkbox";input.checked=value;input.addEventListener("change",()=>{set(input.checked);this.#draw();});row.append(input,document.createTextNode(label));controls.append(row);
    }
    main.append(controls,el("p","gaze-hint","Guarda la webcam mantenendo il volto visibile. Le frecce rappresentano la direzione, non il punto sul monitor."));
    const side=el("aside","gaze-readings");side.append(el("h3","","Sguardo in tempo reale"));this.#stats=el("dl","gaze-metrics");side.append(this.#stats);
    for(const label of ["Orizzontale · yaw","Verticale · pitch","Inclinazione testa · roll","Stime al secondo"]){const value=el("dd","","—");this.#metricValues.push(value);this.#stats.append(el("dt","",label),value);}
    const details=el("details","gaze-details");details.append(el("summary","","Dettagli tecnici"));this.#detailsBox=details;this.#details=el("pre");details.append(this.#details);details.addEventListener("toggle",()=>this.#showDetails());side.append(details);layout.append(main,side);this.#root.append(header,layout);container.replaceChildren(this.#root);
    this.#stream=new GazeStream(this.#o,"GazeDirection",s=>this.update(s),s=>this.#status.textContent=s);this.#metrics(null);
  }
  attach() {
    if (this.#timer || this.#released) return;
    this.#stream.attach(); this.#showVideo();
    this.#timer=setInterval(()=>{if(this.#last && performance.now()-this.#last>1500){this.#sample=null;this.#status.textContent="Nessun dato recente";this.#draw();this.#metrics(null);}},400);
  }
  #showVideo() {
    if(this.#videoShown || this.#released)return;
    const source=this.#o.inputSources?.VideoStream;
    if(source && this.#canvas.transferControlToOffscreen) {
      this.#o.renderer.show(this.#videoId,[source],this.#canvas.transferControlToOffscreen(),{}, {onState:(s,d)=>{if(s==="error")this.#status.textContent=d;}});this.#videoShown=true;
    } else this.#status.textContent="Collega il video alla porta VideoStream";
  }
  setVisible(visible) {
    this.#stream?.setVisible(visible); this.setPaused(!visible);
    if(visible)this.#showVideo();
    else if(this.#videoShown) {
      this.#o.renderer.hide(this.#videoId);this.#videoShown=false;
      const fresh=el("canvas","gaze-video");this.#canvas.replaceWith(fresh);this.#canvas=fresh;
    }
  }
  update(sample) {
    if(this.#released || this.#paused || !this.#alive || sample.schema!=="gaze-direction.v1")return;
    const now=performance.now();if(this.#last && now>this.#last)this.#rate=this.#rate ? .8*this.#rate+.2*1000/(now-this.#last) : 1000/(now-this.#last);this.#last=now;
    this.#sample=sample;this.#status.textContent=statusText(sample.status);this.#showDetails();this.#metrics(sample);this.#draw();
  }
  #metrics(s) {
    const angle=v=>Number.isFinite(v)?v.toFixed(1)+"°":"—";
    const values=[angle(s?.yaw_deg),angle(s?.pitch_deg),angle(s?.roll_deg),s?this.#rate.toFixed(1):"—"];
    values.forEach((value,i)=>{if(this.#metricValues[i].textContent!==value)this.#metricValues[i].textContent=value;});
  }
  #showDetails() { if(this.#detailsBox?.open && this.#sample)this.#details.textContent=JSON.stringify(this.#sample,null,2); }
  #draw() {
    this.#svg.replaceChildren();const s=this.#sample;
    if(s?.status!=="ok" || !Array.isArray(s.image_size_px) || !s.image_size_px.every(v=>Number.isFinite(v)&&v>0) || !Array.isArray(s.bbox_xyxy_px) || s.bbox_xyxy_px.length!==4 || !s.bbox_xyxy_px.every(Number.isFinite) || !Number.isFinite(s.yaw_rad) || !Number.isFinite(s.pitch_rad))return;
    const [w,h]=s.image_size_px;this.#svg.setAttribute("viewBox",`0 0 ${w} ${h}`);this.#svg.style.aspectRatio=`${w}/${h}`;this.#canvas.parentElement.style.aspectRatio=`${w}/${h}`;
    const [x1,y1,x2,y2]=s.bbox_xyxy_px;const stroke=Math.max(2,w/480);
    const shape=(tag,attrs)=>{const n=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));this.#svg.append(n);return n;};
    if(this.#box)shape("rect",{x:x1,y:y1,width:x2-x1,height:y2-y1,rx:12,fill:"none",stroke:"#45e5bc","stroke-width":stroke});
    const eyes=Array.isArray(s.eyes_px)?s.eyes_px.filter(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)):[];
    if(this.#eyes)for(const [x,y] of eyes)shape("circle",{cx:x,cy:y,r:stroke*2,fill:"#45e5bc"});
    if(this.#arrows)for(const [x,y] of eyes.length?eyes:[[(x1+x2)/2,(y1+y2)/2]]) {
      const length=(x2-x1)*1.3, dx=Math.sin(s.yaw_rad)*length,dy=-Math.sin(s.pitch_rad)*length;
      const ex=x+dx,ey=y+dy,a=Math.atan2(dy,dx),head=stroke*4;
      shape("path",{d:`M${x},${y} L${ex},${ey} M${ex-head*Math.cos(a-.5)},${ey-head*Math.sin(a-.5)} L${ex},${ey} L${ex-head*Math.cos(a+.5)},${ey-head*Math.sin(a+.5)}`,fill:"none",stroke:"#ffe29a","stroke-width":stroke*1.4,"stroke-linecap":"round"});
    }
  }
  setAlive(alive) {this.#alive=alive;if(!alive){this.#sample=null;this.#status.textContent="Nodo scollegato";this.#draw();this.#metrics(null);}}
  setPaused(paused) {this.#paused=paused;this.#stream.pause(paused);if(this.#videoShown)this.#o.renderer.control(this.#videoId,"pause",{paused});if(paused){this.#sample=null;this.#draw();}}
  release() {this.#released=true;clearInterval(this.#timer);this.#stream?.release();if(this.#videoShown)this.#o.renderer.hide(this.#videoId);this.#svg?.replaceChildren();}
}

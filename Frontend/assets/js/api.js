/* The only file that talks to the backend. Every number on screen comes from the ML-backed API (BASE_URL). */
const BASE_URL='/api';
async function http(path,opts={}){
 const o={...opts,headers:{'Content-Type':'application/json',...(opts.headers||{})}};
 if(o.body&&typeof o.body!=='string')o.body=JSON.stringify(o.body);
 let r;try{r=await fetch(BASE_URL+path,o)}catch(e){const x=new Error('Cannot reach the server. Is the backend running?');x.status=0;throw x}
 let d=null;try{d=await r.json()}catch(e){}
 if(!r.ok){const x=new Error((d&&d.error&&d.error.message)||'Request failed ('+r.status+')');x.status=r.status;x.code=d&&d.error&&d.error.code;throw x}
 return d}
const qs=o=>{const p=Object.entries(o).filter(([,v])=>v!==undefined&&v!==null&&v!=='').map(([k,v])=>k+'='+encodeURIComponent(v)).join('&');return p?'?'+p:''};
const api={
 getOverview:()=>http('/overview'),
 getFeed:()=>http('/transactions'),
 getStream:seq=>http('/stream?seq='+seq),
 getCases:(f={})=>http('/cases'+qs(f)),
 getCaseCount:()=>http('/cases/count'),
 getCase:id=>http('/cases/'+encodeURIComponent(id)),
 applyAction:(id,a)=>http('/cases/'+encodeURIComponent(id)+'/action',{method:'POST',body:{a}}),
 askAssistant:(id,q,language='en')=>http('/cases/'+encodeURIComponent(id)+'/ask',{method:'POST',body:{q,language}}),
 getGraph:(eid,depth)=>eid?http('/graph/'+encodeURIComponent(eid)+qs({depth})):http('/graph'+qs({depth})),
 getScenarios:()=>http('/demo/scenarios'),
 scoreTransaction:t=>http('/score',{method:'POST',body:t})};
const showError=(el,e)=>{el.innerHTML=`<div class=card><h2>Something went wrong</h2><p class=mut>${esc(e.message||String(e))}</p><p><a href="index.html">Return to the dashboard</a></p></div>`};

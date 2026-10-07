layout('Dashboard');const $=s=>document.querySelector(s);let rows=[],seq=0,run=true,sec=0,failed=false;
function countUp(el,to,fmt){const t0=performance.now();(function f(n){const p=Math.min(1,(n-t0)/1100);el.textContent=fmt(to*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(f)})(t0)}
api.getOverview().then(o=>{const k=o.kpis;
 const items=[['Transactions scored',k.scored,x=>Math.round(x).toLocaleString('en-IN'),'Average latency '+k.latency],['Alerts raised',k.alerts,Math.round,'Legit customers with friction '+k.fp],['Blocked or held',k.blocked,Math.round,'Stopped before money moved'],['Value stopped',k.protected,BDT,'Held, blocked or frozen']];
 $('#kpis').innerHTML=items.map((x,i)=>`<div class="card kpi" style="--i:${i}"><div class=t>${x[0]}<i class=arr>↗</i></div><b>0</b><small>${x[3]}</small></div>`).join('');
 items.forEach((x,i)=>countUp($('#kpis').children[i].querySelector('b'),x[1],x[2]));
 Chart.defaults.font.family="'Plus Jakarta Sans',sans-serif";
 const tr=o.trend.length?o.trend:[0],mx=Math.max(...tr,1),top=[...tr].sort((a,b)=>b-a).slice(0,3);
 $('#bars').innerHTML=tr.map((v,i)=>{const r=top.indexOf(v);return `<div><i class="${['b1 pk','b2','b3'][r]||'hh'}" data-v="${v} alerts" style="height:${Math.max(14,v/mx*100)}%;animation-delay:${i*50}ms"></i>${esc((o.trend_labels||[])[i]||i)}</div>`}).join('');
 $('#rsn').innerHTML=o.reasons.map((r,i)=>`<div class=li><div class=d>${i+1}</div><b>${esc(r[0])}</b><em>${r[1]}</em></div>`).join('')||'<p class=mut>No alerts yet.</p>';
 const tot=Object.values(o.dist).reduce((a,b)=>a+b,0)||1,hi=(o.dist.high||0)+(o.dist.critical||0);
 $('#gv').innerHTML=Math.round(hi/tot*100)+'%<small>High or critical</small>';
 $('#lg').innerHTML=Object.keys(o.dist).map(l=>`<span style="--c:${RC[l]}">${riskLabel(l)}</span>`).join('');
 new Chart($('#c1'),{type:'doughnut',data:{labels:Object.keys(o.dist).map(riskLabel),datasets:[{data:Object.values(o.dist),backgroundColor:Object.keys(o.dist).map(l=>RC[l]),borderWidth:3,borderRadius:6}]},options:{rotation:-90,circumference:180,cutout:'64%',aspectRatio:1.8,plugins:{legend:{display:false}}}});
 $('#agents').innerHTML=o.agents.map(a=>`<div class="ag"><b>${esc(a.id)}</b><div class="track"><div class="fill" style="background:${RC[riskLvl(a.score)]}" data-w="${a.score*100}"></div></div><span>${a.score.toFixed(2)}</span></div>`).join('')+`<p class="mut">Agent risk vs. peer median ${o.agent_median.toFixed(2)} (last 7 days)</p>`;
 setTimeout(()=>document.querySelectorAll('.fill').forEach(f=>f.style.width=f.dataset.w+'%'),100)}).catch(e=>showError($('#main'),e));
api.getCases().then(cs=>{const c=cs.find(c=>c.severity==='critical')||cs[0];if(c)$('#alert').innerHTML=`<h2>Priority alert</h2>${badge(c.severity)}<h3 class=at>${c.alert_type.replace(/_/g,' ')}</h3><p class=mut>${esc(c.what_happened.slice(0,110))}…</p><a class="btn dark" href="case.html?id=${c.case_id}">Investigate ${esc(c.case_id)}</a>`;else $('#alert').innerHTML='<h2>Priority alert</h2><p class=mut>No open alerts.</p>'}).catch(()=>{});
setInterval(()=>{if(run){sec++;$('#clk').textContent=new Date(sec*1000).toISOString().slice(11,19)}},1000);
$('#pz').onclick=e=>{run=!run;e.target.textContent=run?'❚❚':'▶';$('#st').textContent=run?'Scoring transactions live':'Paused'};
$('#rs').onclick=()=>{run=false;sec=0;seq=0;rows=[];feed();$('#clk').textContent='00:00:00';$('#pz').textContent='▶';$('#st').textContent='Stopped. Press ▶ to restart the feed'};
function feed(first){const f=$('#fr').value;$('#feed').innerHTML=rows.filter(t=>f==='all'||t.risk_level===f).slice(0,6).map((t,i)=>`<div class="tx ${first&&i===0?'new':''} ${t.risk_level}" data-id="${esc(t.id)}" tabindex=0><div class=a>${esc(t.sender).slice(-3)}</div><div><b>${BDT(t.amount_bdt)} · ${t.type.replace('_',' ')}${t.type==='payment'||t.purpose==='betting'?' · '+esc(t.purpose_label):''}</b><small>${esc(t.sender)} → ${esc(t.receiver)} · ${fmtTime(t.timestamp)}</small></div><div class=r>${badge(t.risk_level)}<small>${DEC[t.decision]}</small></div></div>`).join('')||'<p class=mut>No transactions at this risk level yet.</p>'}
$('#fr').onchange=()=>feed();
$('#feed').onclick=e=>{const tr=e.target.closest('.tx');if(!tr||!tr.dataset.id)return;const t=rows.find(x=>x.id===tr.dataset.id);if(!t)return;t.case_id?location.href='case.html?id='+t.case_id:toast(`${DEC[t.decision]}: ${t.reasons[0]?.label||'low-risk pattern'}. No case opened.`)};
/* the feed is a replay of real held-out transactions, each scored by the model on the server */
api.getFeed().then(f=>{rows=f.items.slice(0,40);seq=f.next_seq;feed();
 setInterval(()=>{if(!run||failed)return;api.getStream(seq).then(r=>{seq=r.seq+1;rows.unshift(r.txn);rows=rows.slice(0,40);feed(true)}).catch(e=>{failed=true;$('#st').textContent='Feed stopped: '+e.message})},CFG().speed)}).catch(e=>{$('#feed').innerHTML=`<p class=mut>${esc(e.message)}</p>`});
function queue(){api.getCases({alert_type:$('#fa').value,limit:200}).then(cs=>{const l=cs.filter(c=>c.severity==='critical'||c.severity==='high').slice(0,5);$('#q').innerHTML=l.map(c=>`<div class="qrow">${badge(c.severity)}<span><b>${esc(c.case_id)}</b> · ${c.alert_type.replace(/_/g,' ')} · ${esc(c.purpose_label)}<br><span class=mut>${esc(c.what_happened.slice(0,90))}…</span></span><a class="btn" href="case.html?id=${c.case_id}">Investigate</a></div>`).join('')||'<p class=mut>No open alerts of this type. Pick another type.</p>'}).catch(e=>{$('#q').innerHTML=`<p class=mut>${esc(e.message)}</p>`})}
$('#fa').onchange=queue;queue();

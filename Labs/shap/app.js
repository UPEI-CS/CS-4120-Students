'use strict';
const $=id=>document.getElementById(id);
// Rows index complaints (0 or 3); columns index tenure (24 or 2 months).
const fittedOutputs=[[.10,.10],[.30,.70]];
const reference=[[0,0],[1,0],[0,1],[1,1]];
const predict=(c,t)=>fittedOutputs[c][t];
function value(c,t,keepC,keepT){return reference.reduce((s,r)=>s+predict(keepC?c:r[0],keepT?t:r[1]),0)/reference.length;}
function explain(c,t){const base=value(c,t,false,false),onlyC=value(c,t,true,false),onlyT=value(c,t,false,true),prediction=predict(c,t);return {base,onlyC,onlyT,prediction,c:((onlyC-base)+(prediction-onlyT))/2,t:((onlyT-base)+(prediction-onlyC))/2};}
const fmt=x=>Number((x*100).toFixed(1)).toString();
const signed=x=>(x>=0?'+':'−')+fmt(Math.abs(x));
let revealed=0;
function selection(){return [Number($('complaints').value),Number($('tenure').value)];}
function draw(){const [c,t]=selection(),e=explain(c,t);$('prediction').textContent=fmt(e.prediction)+'%';
 const px=x=>300+x*760;let svg='';
 for(let tick=0;tick<=.8+1e-6;tick+=.1){let x=px(tick);svg+=`<line x1="${x}" y1="35" x2="${x}" y2="300" stroke="#e2e8eb"/><text x="${x}" y="325" text-anchor="middle" fill="#526575" font-size="15">${Math.round(tick*100)}%</text>`;}
 svg+='<text x="300" y="20" fill="#526575" font-size="15">Running predicted probability of leaving</text>';
 svg+=`<text x="12" y="75" fill="#203b50" font-size="20">Reference average</text><rect x="300" y="50" width="${e.base*760}" height="32" rx="4" fill="#274a63"/><text x="${px(e.base)+8}" y="74" fill="#274a63" font-size="21">${fmt(e.base)}%</text>`;
 let running=e.base;
 const terms=[{name:`Complaints: ${c?3:0} this month`,v:e.c},{name:`Tenure: ${t?2:24} months`,v:e.t}];
 terms.forEach((term,i)=>{let y=125+i*65;svg+=`<text x="12" y="${y+22}" fill="#203b50" font-size="19">${term.name}</text>`;if(i<revealed){let next=running+term.v,positive=term.v>=0,color=positive?'#75477e':'#39766f',start=px(running),end=px(next);svg+=`<line x1="${start}" y1="${y-25}" x2="${start}" y2="${y+33}" stroke="#87999f" stroke-dasharray="4 4"/><rect x="${Math.min(start,end)}" y="${y}" width="${Math.max(Math.abs(term.v)*760,2)}" height="32" rx="4" fill="${color}"/><text x="${Math.max(start,end)+10}" y="${y+22}" fill="${color}" font-size="21">${signed(term.v)} points</text>`;running=next;}else svg+=`<text x="300" y="${y+22}" fill="#687e8b" font-size="18">Reveal this contribution</text>`;});
 svg+=`<text x="12" y="282" fill="#203b50" font-size="20">${revealed===2?'Final prediction':'Current total'}</text><line x1="${px(running)}" y1="254" x2="${px(running)}" y2="290" stroke="#274a63" stroke-width="4"/><text x="${px(running)+12}" y="282" fill="#274a63" font-size="23">${fmt(running)}%</text>`;
 $('waterfall').innerHTML=svg;
 $('readout').textContent='30%'+(revealed>=1?' '+signed(e.c)+' points':'')+(revealed===2?' '+signed(e.t)+' points':'')+' = '+fmt(running)+'%';
 $('meaning').textContent=revealed===0?'The starting point is the model’s average prediction for the four reference customers.':revealed===1?`For this customer, complaints ${e.c>=0?'raises':'lowers'} the prediction by ${fmt(Math.abs(e.c))} percentage points relative to the reference explanation.`:`Tenure ${e.t>=0?'raises':'lowers'} it by ${fmt(Math.abs(e.t))} points. The contributions now add up to the model’s ${fmt(e.prediction)}% prediction.`;
 $('next').textContent=['Show complaints contribution','Show tenure contribution','Explanation complete'][revealed];$('next').disabled=revealed===2;drawOrders(e);}
function drawOrders(e){const order=Number($('order').value),first=order?e.onlyT:e.onlyC;const name=order?'Tenure':'Complaints',second=order?'Complaints':'Tenure';$('order-flow').innerHTML=`<div class="flow"><span>Reference<br>30%</span><span>${name} supplied<br>${fmt(first)}%</span><span>Both supplied<br>${fmt(e.prediction)}%</span></div><p>${name} adds ${signed(first-e.base)} points. ${second} adds ${signed(e.prediction-first)} points.</p>`;$('averages').innerHTML=`<p>Complaints: (${signed(e.onlyC-e.base)} + ${signed(e.prediction-e.onlyT)}) / 2 = ${signed(e.c)} points.</p><p>Tenure: (${signed(e.onlyT-e.base)} + ${signed(e.prediction-e.onlyC)}) / 2 = ${signed(e.t)} points.</p>`;$('coalitions').innerHTML=`<table><thead><tr><th>Customer information supplied</th><th>Average model prediction</th></tr></thead><tbody><tr><td>Neither feature</td><td>30%</td></tr><tr><td>Complaints only</td><td>${fmt(e.onlyC)}%</td></tr><tr><td>Tenure only</td><td>${fmt(e.onlyT)}%</td></tr><tr><td>Both features</td><td>${fmt(e.prediction)}%</td></tr></tbody></table>`;}
function reset(){revealed=0;draw();}$('next').onclick=()=>{revealed=Math.min(2,revealed+1);draw();};$('reset').onclick=reset;['complaints','tenure'].forEach(id=>$(id).onchange=reset);$('order').onchange=draw;reset();

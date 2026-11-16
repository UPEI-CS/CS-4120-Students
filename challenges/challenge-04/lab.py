"""CS-4120 Challenge 4. Public-data model search and exact finite-background SHAP.
No test-period scoring, no external service, no private records.
"""
import hashlib, io, json, math, platform, time, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
FEATURES = ['recency_days', 'orders', 'spend', 'units', 'products']
SEEDS = (4120, 4121, 4122, 4123, 4124)
BACKGROUND_SEED = 2026
BACKGROUND_SIZE = 32
DATA_URL = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
QUESTIONS = [
'What is a candidate in this search? Give two actual candidates from your run and explain the difference between them.',
'Why search instead of training only one configuration? Use the validation leaderboard to justify which candidate the selection rule chose.',
'For one supplied customer, what does the SHAP plot say about the prediction, and what does it not establish about the real customer?',
'Across the fixed random-forest configuration’s seeds, what changed or stayed stable in performance and SHAP, and how strongly can you defend the feature story?'
]
def frame_hash(frame):
    return hashlib.sha256(np.asarray(frame, dtype='<f8').tobytes()).hexdigest()
def load_public_data(path='online_retail_II.zip', allow_download=True):
    path=Path(path)
    if not path.exists():
        if not allow_download: raise FileNotFoundError('Upload online_retail_II.zip or enable the official public-data download.')
        temporary=path.with_name(path.name+'.part')
        urllib.request.urlretrieve(DATA_URL, temporary)
        with zipfile.ZipFile(temporary) as archive:
            if 'online_retail_II.xlsx' not in archive.namelist() or archive.testzip() is not None: raise ValueError('Official download is incomplete; retry Prepare public data.')
        temporary.replace(path)
    raw_bytes=path.read_bytes()
    if not zipfile.is_zipfile(io.BytesIO(raw_bytes)): raise ValueError('The uploaded online_retail_II.zip is incomplete or invalid. Replace it with the official ZIP in Colab Files, then retry')
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        sheets=pd.read_excel(io.BytesIO(z.read('online_retail_II.xlsx')), sheet_name=None, engine='openpyxl')
    raw=pd.concat(sheets.values(), ignore_index=True).rename(columns={'Customer ID':'customer','CustomerID':'customer','Invoice':'invoice','InvoiceNo':'invoice','Price':'price','UnitPrice':'price','InvoiceDate':'date','Quantity':'quantity','StockCode':'product'})
    raw['date']=pd.to_datetime(raw.date)
    clean=raw.drop_duplicates().dropna(subset=['customer','date']).copy()
    clean=clean[(clean.quantity>0)&(clean.price>0)&~clean.invoice.astype(str).str.upper().str.startswith('C')].copy()
    clean['customer']=clean.customer.astype('int64').astype(str)
    clean['value']=clean.quantity*clean.price
    return clean, {'kind':'real-data','archive_sha256':hashlib.sha256(raw_bytes).hexdigest(),'raw_rows':len(raw),'clean_rows':len(clean)}
def snapshot(data, cutoff, lookback=90, horizon=60):
    cutoff=pd.Timestamp(cutoff); end=cutoff+pd.Timedelta(days=horizon)
    if end>data.date.max(): raise ValueError('Incomplete outcome window')
    before=data[(data.date>=cutoff-pd.Timedelta(days=lookback))&(data.date<cutoff)]
    after=data[(data.date>=cutoff)&(data.date<end)]
    out=before.groupby('customer').agg(last=('date','max'),orders=('invoice','nunique'),spend=('value','sum'),units=('quantity','sum'),products=('product','nunique'))
    out['recency_days']=(cutoff-out.pop('last')).dt.total_seconds()/86400
    out['no_purchase_60d']=(~out.index.isin(after.customer.unique())).astype(int)
    return out

def forest(seed):
    return RandomForestClassifier(n_estimators=150,max_depth=6,min_samples_leaf=10,random_state=seed,n_jobs=1)
def model_candidates():
    out=[('Prior baseline','Constant training prevalence',DummyClassifier(strategy='prior'))]
    for c in (.1,1,10):
        out.append((f'Logistic C={c}',f'log1p + training-only scaler; C={c}',make_pipeline(FunctionTransformer(np.log1p,validate=False),StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=SEEDS[0]))))
    for depth in (3,6):
        out.append((f'Forest depth={depth}',f'150 trees; depth={depth}; min leaf=10; seed={SEEDS[0]}',RandomForestClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=10,random_state=SEEDS[0],n_jobs=1)))
    return out
def probability(model, frame):
    classes=np.asarray(model.classes_)
    ix=np.flatnonzero(classes==1)
    if len(ix)!=1: raise ValueError('Training must contain the positive class 1.')
    return model.predict_proba(frame)[:,int(ix[0])]
def score(model, valid):
    p=probability(model,valid[FEATURES]); y=valid.no_purchase_60d.to_numpy()
    n=max(1,int(np.ceil(len(y)*.2)))
    idx=np.argsort(-p,kind='stable')[:n]
    return {'validation_AP':float(average_precision_score(y,p)),'top20_precision':float(y[idx].mean())}
def exact_background_shap(model, cases, background):
    """Exact Shapley values of v(S)=mean_b f(x_S,b_notS).
    Enumerate all 2**5 coalitions. Exact for the chosen empirical background,
    not a conditional distribution estimate or population/causal explanation.
    """
    cases=cases[FEATURES]; background=background[FEATURES]
    m=len(FEATURES); nbg=len(background); masks=range(1<<m)
    blocks=[]
    for row in cases.to_numpy():
        for mask in masks:
            block=background.to_numpy().copy()
            for j in range(m):
                if mask & (1<<j): block[:,j]=row[j]
            blocks.append(block)
    batch=pd.DataFrame(np.vstack(blocks),columns=FEATURES)
    v=probability(model,batch).reshape(len(cases),1<<m,nbg).mean(axis=2)
    weights=[math.factorial(k)*math.factorial(m-k-1)/math.factorial(m) for k in range(m)]
    phi=np.zeros((len(cases),m))
    for j in range(m):
        for mask in masks:
            if not mask & (1<<j):
                phi[:,j]+=weights[mask.bit_count()]*(v[:,mask|(1<<j)]-v[:,mask])
    prediction=probability(model,cases)
    error=np.abs(v[:,0]+phi.sum(axis=1)-prediction)
    if not np.all(error<1e-8): raise AssertionError('SHAP additivity failed')
    return [{'probability':float(prediction[i]),'base_value':float(v[i,0]),'shap':phi[i].tolist(),'additivity_error':float(error[i])} for i in range(len(cases))]
def run_experiment(train, valid, source, progress=lambda message:None):
    for frame in (train,valid):
        if len(frame)<3 or set(frame.no_purchase_60d.unique())!={0,1}: raise ValueError('Each cohort needs at least three rows and both outcomes.')
        if not np.isfinite(frame[FEATURES].to_numpy()).all(): raise ValueError('Non-finite feature values')
    models={}; rows=[]
    for name,config,model in model_candidates():
        progress('Training '+name)
        start=time.perf_counter();model.fit(train[FEATURES],train.no_purchase_60d)
        elapsed=time.perf_counter()-start
        models[name]=model
        rows.append({'name':name,'configuration':config,'fit_seconds':float(elapsed),**score(model,valid)})
    rows.sort(key=lambda r:-r['validation_AP']) # stable declared-candidate-order tie break
    winner_name=rows[0]['name']; winner=models[winner_name]
    reference=models['Forest depth=6']
    order=np.argsort(probability(reference,valid[FEATURES]),kind='stable')
    positions=[int(order[round((len(order)-1)*q)]) for q in (.1,.5,.9)]
    cases=valid.iloc[positions][FEATURES].copy()
    rng=np.random.default_rng(BACKGROUND_SEED)
    bgpos=np.sort(rng.choice(len(train),size=min(BACKGROUND_SIZE,len(train)),replace=False))
    background=train.iloc[bgpos][FEATURES].copy()
    progress('Explaining the selected candidate with exact finite-background SHAP')
    explained=exact_background_shap(winner,cases,background)
    seed_runs=[]
    for seed in SEEDS:
        progress('Fixed forest: training/explaining seed '+str(seed))
        model=reference if seed==SEEDS[0] else forest(seed).fit(train[FEATURES],train.no_purchase_60d)
        seed_runs.append({'seed':seed,**score(model,valid),'explanations':exact_background_shap(model,cases,background)})
    created=datetime.now(timezone.utc).isoformat(timespec='seconds')
    provenance={**source,'created_utc':created,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'background_sha256':frame_hash(background),'cases_sha256':frame_hash(cases),'shap_method':'Exact enumeration of all 32 coalitions; positive-class predicted probability; marginal feature replacement averaged over the same training-background rows. Correlated-feature combinations may be unrealistic. These are Shapley values for that specified game, not causal or conditional effects.','background_note':f'{len(background)} training rows sampled once with seed {BACKGROUND_SEED}; identical rows for every model and seed. Background-size uncertainty is not measured.'}
    result={'schema':'cs4120.challenge04.v1','features':FEATURES,'train_n':len(train),'valid_n':len(valid),'validation_prevalence':float(valid.no_purchase_60d.mean()),'search':rows,'winner':{'name':winner_name,**score(winner,valid),'explanations':explained},'seed_runs':seed_runs,'cases':[{'label':label,'validation_row_position':p,'values':cases.iloc[i].tolist()} for i,(label,p) in enumerate(zip(['Lower reference risk','Middle reference risk','Higher reference risk'],positions))],'provenance':provenance}
    result['run_id']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()[:16]
    return result

def render_run_html(result=None):
    payload=json.dumps(result,allow_nan=False).replace('<','\\u003c') if result is not None else 'null'
    return VIEWER_HTML.replace('__EMBEDDED_RUN__',payload)
def show_evidence(result):
    from IPython.display import display,HTML
    import matplotlib.pyplot as plt
    display(HTML('<h3>Validation candidate leaderboard</h3>'))
    display(pd.DataFrame(result['search'])[['name','configuration','validation_AP','fit_seconds']])
    display(HTML('<p><b>Selected by validation AP:</b> '+result['winner']['name']+'. AP is not accuracy; the test period was not evaluated.</p>'))
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,case,e in zip(axes,result['cases'],result['winner']['explanations']):
        vals=np.array(e['shap'])*100
        ax.barh(FEATURES,vals,color=['#007e87' if v>=0 else '#774668' for v in vals]);ax.axvline(0,color='#142b49',lw=.8)
        ax.set_title(f"{case['label']}\n{e['base_value']:.1%} reference → {e['probability']:.1%} prediction")
        ax.set_xlabel('SHAP contribution (probability percentage points)')
    display(fig);plt.close(fig)
    display(HTML('<h3>Fixed forest across five seeds</h3><p>Same rows, configuration, customers and SHAP background; only training randomness changes. Stable results are valid. Five seeds do not establish a confidence interval.</p>'))
    display(pd.DataFrame([{k:v for k,v in s.items() if k!='explanations'} for s in result['seed_runs']]))
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for i,ax in enumerate(axes):
        a=np.array([s['explanations'][i]['shap'] for s in result['seed_runs']])*100
        for j,f in enumerate(FEATURES):ax.plot(SEEDS,a[:,j],marker='o',label=f)
        ax.axhline(0,color='#142b49',lw=.8);ax.set_xticks(SEEDS);ax.set_title(result['cases'][i]['label']);ax.set_xlabel('Training seed');ax.set_ylabel('SHAP (probability percentage points)')
    axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1));display(fig);plt.close(fig)
    display(HTML('<p>Exact SHAP for the specified finite-background game. Marginal replacement can create unrealistic combinations of correlated features. This explains the model, not causes or calibrated probabilities. Download the evidence viewer for readable tables and customer selection.</p>'))

def open_lab():
    import ipywidgets as w
    from IPython.display import display,HTML,clear_output
    state={}; prepare=w.Button(description='1. Prepare public data',layout=w.Layout(width='230px'));run=w.Button(description='2. Run search + SHAP',disabled=True,layout=w.Layout(width='230px'));export=w.Button(description='3. Export evidence',disabled=True,layout=w.Layout(width='230px'))
    status=w.HTML('Start with the public dataset. Initial download and Excel parsing can take several minutes.')
    output=w.Output(); links=w.Output()
    def prep(_):
        prepare.disabled=True
        try:
            status.value='Preparing real data…'
            data,source=load_public_data(); train=snapshot(data,'2011-06-01');valid=snapshot(data,'2011-08-01')
            state.update(train=train,valid=valid,source=source);run.disabled=False
            with output:
                clear_output(wait=True);display(HTML(f'<h3>Real data ready</h3><p>{len(train):,} training and {len(valid):,} validation customers. Five historical features; 90-day history → 60-day inactivity outcome. Training labels mature July 31; validation labels mature September 30. End timestamps excluded.</p>'))
            status.value='Ready. Run the bounded search and five-seed SHAP comparison.'
        except Exception as exc:
            status.value='Preparation failed: '+str(exc)+'. See the notebook setup fallback; retry is safe.';prepare.disabled=False
    def compute(_):
        if run.disabled:return
        run.disabled=True;export.disabled=True
        try:
            result=run_experiment(state['train'],state['valid'],state['source'],lambda msg:setattr(status,'value',msg))
            state['result']=result
            with output:
                clear_output(wait=True);show_evidence(result)
            export.disabled=False;status.value='Complete. Evidence stays fixed for this run; export the JSON and HTML, then answer the four notebook questions.'
        except Exception as exc:
            status.value='Run failed: '+str(exc)+'. Retry uses the same declared seeds and temporal split.';run.disabled=False
    def save(_):
        if export.disabled:return
        import base64
        result=state['result']; identifier=result['run_id']
        json_bytes=json.dumps(result,indent=2,allow_nan=False).encode();html_bytes=render_run_html(result).encode()
        with links:
            clear_output(wait=True)
            for name,mime,payload in [(f'challenge-04-run-{identifier}.json','application/json',json_bytes),(f'challenge-04-evidence-{identifier}.html','text/html',html_bytes)]:
                encoded=base64.b64encode(payload).decode()
                display(HTML(f'<p><a download="{name}" href="data:{mime};base64,{encoded}">Download {name}</a></p>'))
            display(HTML('<p>Open the downloaded HTML in a browser. It contains this exact run; the JSON can also be imported into the standalone visual lab. Keep these with your Word answer file. Nothing was uploaded or published.</p>'))
    prepare.on_click(prep);run.on_click(compute);export.on_click(save)
    display(HTML('<div style="border-left:5px solid #007e87;padding:14px"><h2>Challenge 4 · Search → SHAP → seeds</h2><p>All computation uses public UCI data. The later test period is untouched.</p></div>'))
    display(prepare,run,export,status,output,links)
    return state

VIEWER_HTML = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Challenge 4 · Evidence lab</title><style>\n:root{--navy:#142b49;--teal:#007e87;--plum:#774668;--muted:#5f6e80;--line:#dce4ed}*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--navy);font:17px/1.55 system-ui,sans-serif}header{padding:32px max(24px,calc((100vw - 1120px)/2));border-bottom:1px solid var(--line)}h1{font-size:clamp(28px,4vw,42px);line-height:1.15;margin:8px 0}h2{font-size:25px}h3{font-size:19px}p{max-width:85ch}main{max-width:1168px;margin:auto;padding:24px}button,select,input{font:inherit}button,select{padding:10px 16px;background:#fff;border:1px solid var(--line);border-radius:8px;color:var(--navy)}button{cursor:pointer}button[aria-selected=true]{background:var(--navy);color:white}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #efb33d;outline-offset:3px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0}.card{border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:16px}.stat{font-size:30px;font-weight:750;color:var(--teal)}.small{font-size:14px;color:var(--muted)}.badge{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--teal)}.notice{border-left:4px solid var(--teal);padding:12px 18px;background:#f0f8fa}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px 9px;vertical-align:top}th{font-size:13px;text-transform:uppercase;letter-spacing:.5px}.scroll{overflow:auto}.barrow{display:grid;grid-template-columns:140px 1fr 75px;gap:10px;align-items:center;margin:10px 0}.track{background:#f1f4f8;height:22px;border-radius:4px;overflow:hidden}.bar{height:100%;background:var(--teal)}.bar.negative{background:var(--plum)}.legend{display:flex;gap:20px;flex-wrap:wrap}.error{color:#a52727}.hidden{display:none!important}footer{border-top:1px solid var(--line);padding:24px;font-size:13px;color:var(--muted)}a{color:var(--teal)}@media(max-width:550px){main{padding:14px}.card{padding:16px}.barrow{grid-template-columns:100px 1fr 60px;font-size:13px}th,td{padding:8px 5px;font-size:14px}}\n</style></head><body><header><div class="badge">CS-4120 · Challenge 4</div><h1>Can you defend the feature story?</h1><p>Compare candidate models, inspect a customer explanation, then separate prediction stability from explanation stability.</p><div id="source" class="notice">No run loaded. This viewer never trains a model. Import the JSON exported by your Colab notebook.</div></header><main><div class="card"><label for="file"><strong>Open a notebook run</strong></label> <input id="file" type="file" accept=".json,application/json"><p class="small">All processing stays in this browser. Nothing is uploaded. Questions and submission instructions are in the notebook; this page displays evidence only.</p><div id="error" role="alert" class="error"></div></div><section id="app" class="hidden"><div class="tabs" role="tablist" aria-label="Evidence panels"><button role="tab" data-tab="search" aria-selected="true">1 · Search</button><button role="tab" data-tab="explain" aria-selected="false">2 · SHAP</button><button role="tab" data-tab="seeds" aria-selected="false">3 · Seeds</button></div><section id="search" role="tabpanel"></section><section id="explain" role="tabpanel" class="hidden"></section><section id="seeds" role="tabpanel" class="hidden"></section><details class="card"><summary>Provenance and interpretation</summary><div id="provenance"></div></details></section></main><footer>Public source: Chen (2012), <a href="https://doi.org/10.24432/C5CG6D">UCI Online Retail II</a>, CC BY 4.0. Harbour Goods is fictional framing. Attribution explains a model relative to a reference, not what causes a customer to purchase.</footer><script id="embedded-run" type="application/json">__EMBEDDED_RUN__</script><script>\n\'use strict\';let run=null;const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"\']/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\',\'"\':\'&quot;\',"\'":\'&#39;\'}[c])),num=(v,n=3)=>Number(v).toFixed(n),pct=v=>(100*Number(v)).toFixed(1)+\'%\';\nfunction table(head,rows){return \'<div class="scroll"><table><thead><tr>\'+head.map(x=>\'<th>\'+esc(x)+\'</th>\').join(\'\')+\'</tr></thead><tbody>\'+rows.map(r=>\'<tr>\'+r.map(v=>\'<td>\'+esc(v)+\'</td>\').join(\'\')+\'</tr>\').join(\'\')+\'</tbody></table></div>\'}\nfunction bars(values){const max=Math.max(...values.map(Math.abs),.001);return values.map((v,i)=>\'<div class="barrow"><span>\'+esc(run.features[i])+\'</span><div class="track"><div class="bar \'+(v<0?\'negative\':\'\')+\'" style="width:\'+Math.min(100,100*Math.abs(v)/max)+\'%"></div></div><strong>\'+(v>=0?\'+\':\'\')+num(v*100,2)+\'</strong></div>\').join(\'\')}\nfunction validate(d){\nconst bad=()=>{throw Error(\'Invalid or incomplete Challenge 4 evidence. Export a fresh JSON from the notebook.\');},finite=Number.isFinite,integer=v=>Number.isInteger(v)&&v>=0,prob=v=>finite(v)&&v>=0&&v<=1,str=v=>typeof v===\'string\'&&v.length>0&&v.length<2000;\nif(!d||d.schema!==\'cs4120.challenge04.v1\'||JSON.stringify(d.features)!==JSON.stringify([\'recency_days\',\'orders\',\'spend\',\'units\',\'products\'])||!integer(d.train_n)||!integer(d.valid_n)||d.train_n<3||d.valid_n<3||!prob(d.validation_prevalence)||!str(d.run_id)||!Array.isArray(d.search)||d.search.length!==6||!Array.isArray(d.cases)||d.cases.length!==3||!Array.isArray(d.seed_runs)||d.seed_runs.length!==5)bad();\nfor(const row of d.search)if(!row||!str(row.name)||!str(row.configuration)||!prob(row.validation_AP)||!finite(row.fit_seconds)||row.fit_seconds<0)bad();\nfor(const c of d.cases)if(!c||!str(c.label)||!integer(c.validation_row_position)||c.validation_row_position>=d.valid_n||!Array.isArray(c.values)||c.values.length!==5||!c.values.every(v=>finite(v)&&v>=0))bad();\nconst p=d.provenance;if(!p||![\'real-data\',\'test-fixture\'].includes(p.kind)||!str(p.created_utc)||!str(p.archive_sha256)||!str(p.background_sha256)||!str(p.cases_sha256)||!str(p.shap_method)||!str(p.background_note)||!p.versions||Array.isArray(p.versions)||typeof p.versions!==\'object\'||!Object.values(p.versions).every(str))bad();\nif(!d.winner||!str(d.winner.name)||!d.search.some(s=>s.name===d.winner.name))bad();\nfor(let i=0;i<5;i++)if(d.seed_runs[i].seed!==4120+i)bad();\nfor(const model of [d.winner,...d.seed_runs]){if(!prob(model.validation_AP)||!prob(model.top20_precision)||!Array.isArray(model.explanations)||model.explanations.length!==3)bad();for(const e of model.explanations){if(!e||!prob(e.probability)||!prob(e.base_value)||!Array.isArray(e.shap)||e.shap.length!==5||!e.shap.every(finite)||!finite(e.additivity_error)||e.additivity_error<0||e.additivity_error>1e-7)bad();if(Math.abs(e.base_value+e.shap.reduce((a,b)=>a+b,0)-e.probability)>1e-7)throw Error(\'Invalid SHAP evidence or failed additivity.\');}}\nreturn d}\n\nfunction load(d){run=validate(d);$(\'error\').textContent=\'\';$(\'app\').classList.remove(\'hidden\');$(\'source\').textContent=(d.provenance.kind===\'real-data\'?(d.provenance.role===\'reference\'?\'Recorded real-data reference run\':\'Actual notebook run\'):\'TEST FIXTURE — not course data\')+\' · \'+d.provenance.created_utc+\' · no test-period outcomes used\';renderSearch();$(\'explain\').innerHTML=\'<h2>Explain one validation customer</h2><p>The same three customers are used throughout this run. They were chosen once using the fixed reference forest, not reselected for each seed.</p><label>Customer example <select id="caseSelect">\'+d.cases.map((c,i)=>\'<option value="\'+i+\'">\'+esc(c.label)+\'</option>\').join(\'\')+\'</select></label><div id="casePanel"></div>\';$(\'caseSelect\').addEventListener(\'change\',()=>renderCase(Number($(\'caseSelect\').value)));renderCase(0);$(\'seeds\').innerHTML=\'<h2>Same configuration. Five training seeds.</h2><p>Random forest: depth 6, 150 trees, minimum leaf 10. Training/validation rows, features, customer examples and SHAP background stay fixed. Only training randomness changes.</p>\'+table([\'Seed\',\'Validation AP\',\'Top-20% precision\'],d.seed_runs.map(s=>[s.seed,num(s.validation_AP),pct(s.top20_precision)]))+\'<label>Compare the same customer <select id="seedCase">\'+d.cases.map((c,i)=>\'<option value="\'+i+\'">\'+esc(c.label)+\'</option>\').join(\'\')+\'</select></label><div id="seedPanel"></div>\';$(\'seedCase\').addEventListener(\'change\',()=>renderSeeds(Number($(\'seedCase\').value)));renderSeeds(0);$(\'provenance\').innerHTML=\'<p><strong>Run ID:</strong> \'+esc(d.run_id)+\'</p><p>Archive SHA-256: <code>\'+esc(d.provenance.archive_sha256)+\'</code></p><p>\'+esc(d.provenance.shap_method)+\'</p><p>\'+esc(d.provenance.background_note)+\'</p><p>Training: 2011-06-01; validation: 2011-08-01. Each uses 90 prior days and a 60-day future outcome. Later test outcomes are not evaluated. Customers may recur between time periods.</p><p>Versions: \'+esc(JSON.stringify(d.provenance.versions))+\'</p><p>Background hash: \'+esc(d.provenance.background_sha256)+\'. Cases hash: \'+esc(d.provenance.cases_sha256)+\'.</p><p>Positive purchases exclude returns, cancellations, missing customer IDs and exact duplicates. This historic UK dataset does not establish present-day or PEI performance. No prediction proves that outreach changes behaviour.</p>\';}\nfunction renderSearch(){const d=run;$(\'search\').innerHTML=\'<h2>Six candidates, one validation rule</h2><div class="grid"><div class="card"><div class="small">Selected by highest validation AP</div><div class="stat">\'+esc(d.winner.name)+\'</div></div><div class="card"><div class="small">Validation AP / baseline AP</div><div class="stat">\'+num(d.winner.validation_AP)+\' / \'+num(d.validation_prevalence)+\'</div></div><div class="card"><div class="small">Training / validation customers</div><div class="stat">\'+esc(d.train_n)+\' / \'+esc(d.valid_n)+\'</div></div></div><p>AP is average precision, a ranking summary across thresholds; it is not accuracy. A constant-score baseline has AP equal to prevalence. Tiny differences do not prove a winner generalizes better.</p>\'+table([\'Candidate\',\'Configuration\',\'Validation AP\',\'Fit seconds\'],d.search.map(s=>[s.name,s.configuration,num(s.validation_AP),num(s.fit_seconds,2)]))+\'<p class="notice">Selection uses validation only. The selected model stays fitted on training only. The seed comparison uses a fixed forest configuration, not the best-looking seed or a fresh search.</p>\';}\nfunction renderCase(i){const e=run.winner.explanations[i],c=run.cases[i];$(\'casePanel\').innerHTML=\'<div class="card"><h3>\'+esc(run.winner.name)+\' · \'+esc(c.label)+\'</h3>\'+table([\'Feature\',\'Observed value\'],run.features.map((f,j)=>[f,num(c.values[j],2)]))+\'<p><strong>Reference average \'+pct(e.base_value)+\' + SHAP contributions = prediction \'+pct(e.probability)+\'.</strong></p><div class="legend small"><span>Teal: raises predicted non-purchase probability</span><span>Plum: lowers it</span></div>\'+bars(e.shap)+\'<p class="small">Values are probability percentage points. Bar lengths show magnitude; colour and signed number show direction. Exact additivity error: \'+e.additivity_error.toExponential(1)+\'.</p><p>These are exact Shapley values for this finite-background, marginal-replacement game. They are not causes, confidence intervals, calibrated risks, or a unique explanation independent of the chosen reference.</p></div>\';}\nfunction renderSeeds(i){const vals=run.seed_runs.map(s=>s.explanations[i]),aps=run.seed_runs.map(s=>s.validation_AP);$(\'seedPanel\').innerHTML=\'<div class="card"><h3>\'+esc(run.cases[i].label)+\' · feature contributions across seeds</h3><p>Validation AP range: <strong>\'+num(Math.min(...aps))+\'–\'+num(Math.max(...aps))+\'</strong>. Customer prediction range: <strong>\'+pct(Math.min(...vals.map(x=>x.probability)))+\'–\'+pct(Math.max(...vals.map(x=>x.probability)))+\'</strong>.</p>\'+table([\'Feature\',...run.seed_runs.map(s=>\'Seed \'+s.seed),\'Min / max (pp)\'],run.features.map((f,j)=>{const x=vals.map(e=>100*e.shap[j]);return [f,...x.map(v=>(v>=0?\'+\':\'\')+num(v,2)),num(Math.min(...x),2)+\' / \'+num(Math.max(...x),2)]}))+\'<p>Each cell is a signed SHAP contribution in probability percentage points. Compare magnitudes and signs, not just which feature ranks first.</p>\'+table([\'Seed\',\'Top feature by |SHAP|\',\'Prediction\',\'Reference\'],run.seed_runs.map(s=>{const e=s.explanations[i],k=e.shap.map(Math.abs).indexOf(Math.max(...e.shap.map(Math.abs)));return [s.seed,run.features[k],pct(e.probability),pct(e.base_value)]}))+\'<p class="notice">Five seeds describe variability here, not a confidence interval or proof of robustness. Stable explanations are a valid result. Different seeds can change the reference prediction as well as contributions even though the reference rows are fixed.</p></div>\';}\ndocument.querySelectorAll(\'[data-tab]\').forEach(b=>b.addEventListener(\'click\',()=>{for(const t of [\'search\',\'explain\',\'seeds\'])$(t).classList.toggle(\'hidden\',t!==b.dataset.tab);document.querySelectorAll(\'[data-tab]\').forEach(x=>x.setAttribute(\'aria-selected\',String(x===b)));}));$(\'file\').addEventListener(\'change\',async e=>{try{const f=e.target.files[0];if(!f)return;if(f.size>5000000)throw Error(\'File too large. Select the small notebook run JSON.\');load(JSON.parse(await f.text()));}catch(err){$(\'error\').textContent=err.message;}});try{const initial=JSON.parse($(\'embedded-run\').textContent);if(initial)load(initial);}catch(err){$(\'error\').textContent=err.message;}\n</script></body></html>'

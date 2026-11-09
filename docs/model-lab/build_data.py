"""Rebuild the synthetic CV table and exact interventional Shapley explanations.
Run from this directory: python3 build_data.py
Dependencies: numpy pandas scikit-learn matplotlib.
All 32 feature subsets are enumerated using a fixed 60-row development background.
This is a small teaching implementation, not a scalable replacement for TreeSHAP.
"""
import os
os.environ['MPLBACKEND']='Agg'
import sys, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
display = lambda *args: None
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.inspection import permutation_importance
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 115})
FIG = Path('/tmp/cs4120-figures'); FIG.mkdir(exist_ok=True)
TEAL, BLUE, ORANGE = '#16796D', '#2467A6', '#C76628'
def finish(name):
    plt.tight_layout(); plt.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight'); plt.show()
print('Python', sys.version.split()[0], '| scikit-learn', sklearn.__version__)
rng = np.random.default_rng(4120)
n = 1500
latent = rng.normal(size=n)
complaints = rng.normal(size=n)
tenure = rng.normal(size=n)
X = pd.DataFrame({'Recency': latent + rng.normal(0, .08, n),
                  'Inactivity': latent + rng.normal(0, .08, n),
                  'Complaints': complaints, 'Tenure': tenure,
                  'Noise': rng.normal(size=n)})
# A deliberately strong, noisy predictive signal. All columns are standardised synthetic units.
p = 1 / (1 + np.exp(-(3.5 * latent + 1.8 * complaints - .6 * tenure)))
PROXY_NOISE = 0.0  # Change to 0.08 for an imperfect proxy, then rerun all cells.
X['Inactivity'] = X['Recency'] + PROXY_NOISE * rng.normal(size=n)
y = pd.Series(rng.binomial(1, p), name='leave')
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=.4, stratify=y, random_state=17)
X_audit, X_test, y_audit, y_test = train_test_split(X_hold, y_hold, test_size=.5,
                                                stratify=y_hold, random_state=19)
print('Development / audit / test:', len(X_dev), len(X_audit), len(X_test))
print('Recency-inactivity correlation:', round(X_dev.corr().loc['Recency','Inactivity'],3))
plt.figure(figsize=(7,4)); plt.scatter(X_dev.Recency, X_dev.Inactivity, c=y_dev,
    cmap='coolwarm', s=12, alpha=.5); plt.xlabel('Recency'); plt.ylabel('Inactivity')
plt.title(f'Two inputs; PROXY_NOISE = {PROXY_NOISE:.2f}'); finish('correlated_inputs')
feature_options = [list(X.columns), ['Recency','Complaints','Tenure','Noise'],
                   ['Inactivity','Complaints','Tenure','Noise'], ['Complaints','Tenure','Noise']]
option_names = ['Both signals','Recency only','Inactivity only','Neither signal']
depths, leaves = [2,3,4,5], [5,15,30]
configs = list(itertools.product(range(4), range(4), range(3)))
folds = list(StratifiedKFold(3, shuffle=True, random_state=23).split(X_dev,y_dev))
scoring = {'precision': make_scorer(precision_score, zero_division=0),
           'recall': make_scorer(recall_score, zero_division=0)}
def build_model(config):
    f,d,l = config
    return make_pipeline(ColumnTransformer([('keep','passthrough',feature_options[f])],
                                           remainder='drop'),
                         DecisionTreeClassifier(max_depth=depths[d],
                             min_samples_leaf=leaves[l], random_state=7))
records=[]
for cid, cfg in enumerate(configs):
    cv = cross_validate(build_model(cfg), X_dev, y_dev, cv=folds, scoring=scoring)
    f,d,l=cfg
    records.append({'id':cid, 'features':option_names[f], 'depth':depths[d],
                    'leaf':leaves[l], 'precision':cv['test_precision'].mean(),
                    'recall':cv['test_recall'].mean()})
results=pd.DataFrame(records).set_index('id')
results['eligible']=results.recall >= .70
eligible=results[results.eligible]
assert len(eligible), 'No eligible candidate: inspect the rule and data without using the test set.'
best=float(eligible.precision.max())
winner=int(eligible.precision.idxmax())
print('48 candidate evaluations = 144 CV tree fits. Final refits are additional.')
display(results.sort_values('precision',ascending=False).head(8))
from math import factorial
from pathlib import Path
background=X_dev.sample(n=60,random_state=41)
audit=X_audit.iloc[:100]
models=[]
for cid,cfg in enumerate(configs):
    model=build_model(cfg).fit(X_dev,y_dev)
    # For each coalition, keep present features at this case's values and
    # replace absent features with the corresponding values of background rows.
    # Absent columns are drawn jointly from each background row.
    values=np.zeros((len(audit),32))
    for mask in range(32):
        hybrid=np.tile(background.to_numpy(),(len(audit),1))
        for j in range(5):
            if mask & (1<<j): hybrid[:,j]=np.repeat(audit.iloc[:,j].to_numpy(),len(background))
        values[:,mask]=model.predict_proba(pd.DataFrame(hybrid,columns=X.columns))[:,1].reshape(len(audit),len(background)).mean(axis=1)
    phi=np.zeros((len(audit),5))
    for j in range(5):
        for mask in range(32):
            if not mask&(1<<j):
                k=mask.bit_count();weight=factorial(k)*factorial(4-k)/factorial(5)
                phi[:,j]+=weight*(values[:,mask|(1<<j)]-values[:,mask])
    pred=model.predict_proba(audit)[:,1]
    assert np.allclose(values[:,0]+phi.sum(axis=1),pred,atol=1e-10)
    for j,col in enumerate(X.columns):
        if col not in feature_options[cfg[0]]:assert np.allclose(phi[:,j],0,atol=1e-10)
    row=results.loc[cid]
    models.append(dict(id=cid,cfg=list(cfg),features=row.features,depth=int(row.depth),leaf=int(row.leaf),precision=float(row.precision),recall=float(row.recall),eligible=bool(row.eligible),base=float(values[0,0]),pred=pred[:12].tolist(),shap=phi[:12].tolist(),global_shap=np.abs(phi).mean(axis=0).tolist()))
assert np.allclose(models[19]['pred'],models[31]['pred'])
data=dict(models=models,columns=list(X.columns),cases=audit.iloc[:12].to_numpy().tolist(),background_n=60,audit_n=100,method='Exact interventional Shapley, probability output',versions=dict(numpy=np.__version__,sklearn=sklearn.__version__))
Path('data.js').write_text('window.LAB_DATA = '+json.dumps(data,separators=(',',':'))+';\n')
print('Validated additivity for 4800 explanations and zero contribution for unused inputs.')

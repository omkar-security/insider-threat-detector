import sqlite3, pandas as pd, numpy as np, json, time, gc, sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from anonymiser import anonymise, prepare_log, pseudonymise, features_from_anonymised, ANON_ML_COLS

conn=sqlite3.connect('insider_threat.db')
log=pd.read_sql('SELECT * FROM unified_log',conn)
emps=pd.read_sql('SELECT * FROM employees',conn)
conn.close()
log=prepare_log(log,emps)
emps['pseudonym']=emps['employee_id'].map(pseudonymise)
print('loaded', len(log), flush=True)
rows=[]
for k in [2,5,10,20]:
    t=time.time()
    res=anonymise(log,k)
    feats=features_from_anonymised(res['data']).merge(emps[['pseudonym','is_insider']],on='pseudonym')
    X,y=feats[ANON_ML_COLS].fillna(0),feats['is_insider']
    iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=200).fit(X)
    preds=(iso.predict(X)==-1).astype(int); scores=iso.decision_function(X)
    r=dict(k=k,achieved_k=res['verification']['achieved_k'],levels=str(res['levels']),
        ncp=round(res['ncp'],4),supp=round(res['suppression_rate'],4),n_suppressed=res['n_suppressed'],
        prec=round(precision_score(y,preds,zero_division=0),3),rec=round(recall_score(y,preds,zero_division=0),3),
        f1=round(f1_score(y,preds,zero_division=0),3),auc=round(roc_auc_score(y,-scores),4),
        n_classes=res['verification']['n_classes'],mean_cls=round(res['verification']['mean_class_size'],1),
        n_entities=int(feats.shape[0]),tp=int(((preds==1)&(y==1)).sum()),fp=int(((preds==1)&(y==0)).sum()),
        fn=int(((preds==0)&(y==1)).sum()))
    rows.append(r); print(k,'done in',round(time.time()-t,1),'s',r,flush=True)
    json.dump(rows, open('tradeoff_out.json','w'), indent=2)
    del res,feats,X,y,iso; gc.collect()
print('ALL DONE',flush=True)

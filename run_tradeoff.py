import sqlite3, pandas as pd, numpy as np, json, time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from anonymiser import anonymise, prepare_log, pseudonymise, features_from_anonymised, ANON_ML_COLS

t0=time.time()
conn=sqlite3.connect('insider_threat.db')
log=pd.read_sql('SELECT * FROM unified_log',conn)
emps=pd.read_sql('SELECT * FROM employees',conn)
conn.close()
log=prepare_log(log,emps)
emps['pseudonym']=emps['employee_id'].map(pseudonymise)
rows=[]
for k in [2,5,10,20]:
    res=anonymise(log,k)
    feats=features_from_anonymised(res['data']).merge(emps[['pseudonym','is_insider']],on='pseudonym')
    X,y=feats[ANON_ML_COLS].fillna(0),feats['is_insider']
    iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=200).fit(X)
    preds=(iso.predict(X)==-1).astype(int)
    scores=iso.decision_function(X)
    rows.append(dict(k=k,achieved_k=res['verification']['achieved_k'],levels=str(res['levels']),
        ncp=round(res['ncp'],4),supp=round(res['suppression_rate'],4),
        n_suppressed=res['n_suppressed'],
        prec=round(precision_score(y,preds,zero_division=0),3),
        rec=round(recall_score(y,preds,zero_division=0),3),
        f1=round(f1_score(y,preds,zero_division=0),3),
        auc=round(roc_auc_score(y,-scores),4),
        n_classes=res['verification']['n_classes'],
        mean_cls=round(res['verification']['mean_class_size'],1),
        n_detected=int(feats.shape[0])))
    json.dump(rows, open('tradeoff_out.json','w'), indent=2)
json.dump({'rows':rows,'elapsed':round(time.time()-t0,1),'done':True}, open('tradeoff_out.json','w'), indent=2)

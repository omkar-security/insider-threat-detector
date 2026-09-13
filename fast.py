import sqlite3, pandas as pd, numpy as np, json, time
import anonymiser as A
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

t0=time.time()
conn=sqlite3.connect('insider_threat.db')
log=pd.read_sql('SELECT * FROM unified_log',conn)
emps=pd.read_sql('SELECT * FROM employees',conn)
conn.close()
log=A.prepare_log(log,emps)
emps['pseudonym']=emps['employee_id'].map(A.pseudonymise)

# precompute static columns ONCE
ts=pd.to_datetime(log['timestamp'])
hours=ts.dt.hour.values
date=ts.dt.date.astype(str).values
pseud=log['employee_id'].map(A.pseudonymise).values
dept=log['department'].values
sens=log['sensitivity'].values
action=log['action'].values
resource=log['resource'].values
n=len(log)
print('loaded',n,round(time.time()-t0,1),flush=True)

def gen_qids(levels):
    li=levels['identity']
    if li==0: rg=dept
    elif li==1: rg=np.array([A.DEPT_TO_DIVISION[d] for d in dept])
    else: rg=np.full(n,'Any Role')
    lt=levels['time']
    if lt==0: tw=np.char.zfill(hours.astype(str),2)
    elif lt==1: tw=np.array([A._hour_to_window(h) for h in range(24)])[hours]
    elif lt==2: tw=np.where((hours>=6)&(hours<18),'Core','Off')
    else: tw=np.full(n,'Any')
    lr=levels['resource']
    if lr==0: rc=resource
    elif lr==1: rc=sens
    else: rc=np.full(n,'Any')
    la=levels['action']
    if la==0: ac=action
    elif la==1: ac=np.array([A.ACTION_TO_CATEGORY[a] for a in action])
    else: ac=np.full(n,'Any')
    sn=log['seniority'].values
    return rg,sn,tw,rc,ac

rows=[]
for k in [2,5,10,20]:
    tk=time.time()
    chosen=None
    for levels in A._level_lattice():
        rg,sn,tw,rc,ac=gen_qids(levels)
        key=pd.DataFrame({'a':rg,'b':sn,'c':tw,'d':rc,'e':ac})
        sizes=key.assign(_o=1).groupby(['a','b','c','d','e'])['_o'].transform('size').values
        keep=sizes>=k
        supp=1-keep.mean()
        if supp<=0.05:
            chosen=(levels,keep,supp); break
    levels,keep,supp=chosen
    # build anonymised feature set per pseudonym using kept rows
    sub=log[keep].copy()
    sub['pseudonym']=pseud[keep]
    sub['is_high_sens']=(sub['sensitivity']=='High').astype(int)
    sub['is_usb']=(sub['activity_type']=='usb').astype(int)
    la=levels['action']
    if la==0: act_class=sub['action']
    elif la==1: act_class=sub['action'].map(A.ACTION_TO_CATEGORY)
    else: act_class=pd.Series('Any Action',index=sub.index)
    sub['is_transfer']=act_class.isin(['Data Transfer','send','copy','print']).astype(int)
    feats=sub.groupby('pseudonym').agg(
        after_hours_ratio=('is_after_hours','mean'),external_ratio=('is_external','mean'),
        avg_data_volume_kb=('data_volume_kb','mean'),max_data_volume_kb=('data_volume_kb','max'),
        total_data_volume_kb=('data_volume_kb','sum'),high_sensitivity_ratio=('is_high_sens','mean'),
        usb_events=('is_usb','sum'),transfer_ratio=('is_transfer','mean')).reset_index()
    feats['usb_per_day']=feats['usb_events']/180
    feats=feats.merge(emps[['pseudonym','is_insider']],on='pseudonym')
    X=feats[A.ANON_ML_COLS].fillna(0); y=feats['is_insider']
    iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=200).fit(X)
    preds=(iso.predict(X)==-1).astype(int); scores=iso.decision_function(X)
    # achieved k + classes on kept rows
    rg,sn,tw,rc,ac=gen_qids(levels)
    kdf=pd.DataFrame({'a':rg[keep],'b':sn[keep],'c':tw[keep],'d':rc[keep],'e':ac[keep]})
    cs=kdf.groupby(['a','b','c','d','e']).size()
    r=dict(k=k,achieved_k=int(cs.min()),levels=str(levels),ncp=round(A.ncp(levels,log['resource'].nunique()),4),
        supp=round(float(supp),4),n_suppressed=int((~keep).sum()),
        prec=round(precision_score(y,preds,zero_division=0),3),rec=round(recall_score(y,preds,zero_division=0),3),
        f1=round(f1_score(y,preds,zero_division=0),3),auc=round(roc_auc_score(y,-scores),4),
        n_classes=int(len(cs)),mean_cls=round(float(cs.mean()),1),n_entities=int(len(feats)),
        tp=int(((preds==1)&(y==1)).sum()),fp=int(((preds==1)&(y==0)).sum()),fn=int(((preds==0)&(y==1)).sum()))
    rows.append(r); print(k,round(time.time()-tk,1),r,flush=True)
    json.dump(rows,open('tradeoff_out.json','w'),indent=2)
print('DONE',round(time.time()-t0,1),flush=True)

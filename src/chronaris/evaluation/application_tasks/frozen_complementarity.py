"""Subject-nested linear readouts of independent frozen streams and their combinations."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from chronaris.evaluation.application_tasks.grouped_readout_repair import score


def transform_pair(train, test, dimension=None):
    scaler=StandardScaler().fit(train)
    a,b=scaler.transform(train),scaler.transform(test)
    if dimension:
        if dimension>min(a.shape):raise ValueError('insufficient training rows for fixed PCA dimension')
        pca=PCA(dimension,svd_solver='full').fit(a)
        a,b=pca.transform(a),pca.transform(b)
        scaler=StandardScaler().fit(a);a,b=scaler.transform(a),scaler.transform(b)
    return a,b


def predict_prepared(a,y,b,parameter,classes):
    if classes is None:
        return Ridge(alpha=parameter).fit(a,y).predict(b)
    if set(y)!=set(classes):raise ValueError('inner training lacks a class')
    model=LogisticRegression(C=parameter,class_weight='balanced',max_iter=5000,random_state=17).fit(a,y)
    if list(model.classes_)!=list(classes):raise ValueError('probability columns differ from declared classes')
    return model.predict_proba(b)


def decoded(values,classes,bounds=None):
    if classes is not None:return np.asarray(classes)[np.argmax(values,axis=1)]
    return np.clip(values,*bounds) if bounds else values


def grouped_value(y,prediction,groups,classes,bounds=None):
    pred=decoded(prediction,classes,bounds)
    return float(np.mean([score(y[groups==s],pred[groups==s],classes) for s in sorted(set(groups))]))


def evaluate_frozen_combinations(first,second,full,y,groups,train,test,*,classes=None,bounds=None):
    """Outer labels never enter transforms, candidate selection, or returned predictions."""
    first,second,full=[np.asarray(x,float) for x in (first,second,full)]
    y,groups=np.asarray(y),np.asarray(groups)
    train,test=np.asarray(train),np.asarray(test)
    if (any(len(x)!=len(y) or not np.isfinite(x).all() for x in (first,second,full))
        or not len(train) or not len(test) or len(set(train))!=len(train) or len(set(test))!=len(test)
        or set(groups[train])&set(groups[test]) or len(set(groups[train]))<2
        or not np.isfinite(y[train]).all()):raise ValueError('invalid or overlapping nested readout data')
    candidates=(.1,1.,10.) if classes is not None else (.1,1.,10.,100.,1000.,10000.)
    combined=np.concatenate((first,second),axis=1)
    features={'first':(first,None),'second':(second,None),'full':(full,None),
              'concat128':(combined,None),'concat_pca64':(combined,64)}
    inner_cache,final_cache,logs={},{},[]
    for name,(x,dimension) in features.items():
        shape=(len(candidates),len(train),len(classes)) if classes is not None else (len(candidates),len(train))
        oof=np.full(shape,np.nan)
        for subject in sorted(set(groups[train])):
            held=groups[train]==subject
            a,b=transform_pair(x[train[~held]],x[train[held]],dimension)
            for k,parameter in enumerate(candidates):
                oof[k,held]=predict_prepared(a,y[train[~held]],b,parameter,classes)
        if not np.isfinite(oof).all():raise ValueError('incomplete inner predictions')
        inner_cache[name]=oof
        a,b=transform_pair(x[train],x[test],dimension)
        final_cache[name]=[predict_prepared(a,y[train],b,c,classes) for c in candidates]
    rules={'raw':None}
    if bounds is not None:rules['bounded']=bounds
    outputs={}
    for rule,limit in rules.items():
        chosen={}
        for name in features:
            values=[grouped_value(y[train],p,groups[train],classes,limit) for p in inner_cache[name]]
            index=int(np.argmax(values) if classes is not None else np.argmin(values));chosen[name]=index
            outputs[name,rule]=dict(prediction=decoded(final_cache[name][index],classes,limit),
                parameter=candidates[index],weight=None)
            logs.extend(dict(method=name,rule=rule,candidate=i,parameter=c,weight=None,value=v,
                selected=i==index) for i,(c,v) in enumerate(zip(candidates,values)))
            if rule=='raw' and bounds is not None:
                outputs[name,'raw_clipped']=dict(prediction=decoded(final_cache[name][index],classes,bounds),
                    parameter=candidates[index],weight=None)
        a,b=chosen['first'],chosen['second']
        weights=(0.,.25,.5,.75,1.)
        inner=[w*inner_cache['first'][a]+(1-w)*inner_cache['second'][b] for w in weights]
        values=[grouped_value(y[train],p,groups[train],classes,limit) for p in inner]
        best=int(np.argmax(values) if classes is not None else np.argmin(values))
        for name,k in [('equal_prediction',2),('selected_prediction',best)]:
            w=weights[k]
            pred=w*final_cache['first'][a]+(1-w)*final_cache['second'][b]
            outputs[name,rule]=dict(prediction=decoded(pred,classes,limit),
                parameter=[candidates[a],candidates[b]],weight=w)
            if rule=='raw' and bounds is not None:
                outputs[name,'raw_clipped']=dict(prediction=decoded(pred,classes,bounds),
                    parameter=[candidates[a],candidates[b]],weight=w)
        logs.extend(dict(method='selected_prediction',rule=rule,candidate=i,parameter=[candidates[a],candidates[b]],
            weight=w,value=v,selected=i==best) for i,(w,v) in enumerate(zip(weights,values)))
    return outputs,logs

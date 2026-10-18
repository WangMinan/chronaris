"""Inner-subject selection of post-standardization block penalties and bounded heads."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def score(target, prediction, classes=None):
    if classes is not None:
        return float(f1_score(target, prediction, labels=classes, average='macro', zero_division=0))
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def transformed(train, test, candidate):
    projection = candidate.get('projection')
    if projection == 'second24':
        train, test = train[:, 24:48], test[:, 24:48]
    elif projection == 'pca24':
        standard = StandardScaler().fit(train)
        pca = PCA(24, svd_solver='full').fit(standard.transform(train))
        train, test = pca.transform(standard.transform(train)), pca.transform(standard.transform(test))
    standard = StandardScaler().fit(train)
    train, test = standard.transform(train), standard.transform(test)
    weights = candidate.get('weights', (1., 1., 1.))
    if tuple(weights) != (1., 1., 1.):
        if train.shape[1] != 64 or len(weights) != 3 or not np.isfinite(weights).all() or min(weights) <= 0:
            raise ValueError('block penalties require positive weights for 24/24/16 dimensions')
        weights = np.repeat(weights, (24, 24, 16))
        train, test = train * weights, test * weights
    return train, test


def fit_predict(train, target, test, candidate, classes=None):
    a, b = transformed(train, test, candidate)
    if classes is not None:
        if set(target) != set(classes):
            raise ValueError('inner training lacks a classification label; no window-level fallback')
        model = LogisticRegression(C=candidate['parameter'], class_weight='balanced', max_iter=5000, random_state=17)
    else:
        model = Ridge(alpha=candidate['parameter'])
    return model.fit(a, target).predict(b)


def nested_readout(x, y, groups, train, test, candidates, *, classes=None, bounds=None, selections=None):
    """Selections map named arms to candidate indices; all selection uses inner subjects."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    train, test = np.asarray(train), np.asarray(test)
    if (not len(train) or not len(test) or len(set(train)) != len(train) or len(set(test)) != len(test)
        or set(groups[train]) & set(groups[test]) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('invalid readout data or overlapping scoring subjects')
    subjects = sorted(set(groups[train]))
    if len(subjects) < 2:
        raise ValueError('at least two training subjects required for nested selection')
    if bounds is not None and (classes is not None or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
        raise ValueError('invalid bounded regression interval')
    selections = selections or {'selected': list(range(len(candidates)))}
    inner = []
    for subject in subjects:
        a, b = train[groups[train] != subject], train[groups[train] == subject]
        for i, candidate in enumerate(candidates):
            prediction = fit_predict(x[a], y[a], x[b], candidate, classes)
            value = score(y[b], prediction, classes)
            inner.append(dict(subject=str(subject), candidate=i, value=value,
                bounded_value=score(y[b], np.clip(prediction, *bounds)) if bounds is not None else value))
    values = np.array([np.mean([r['value'] for r in inner if r['candidate'] == i]) for i in range(len(candidates))])
    bounded = np.array([np.mean([r['bounded_value'] for r in inner if r['candidate'] == i]) for i in range(len(candidates))])
    outcomes, final = {}, {}
    def predict(i):
        if i not in final:
            final[i] = fit_predict(x[train], y[train], x[test], candidates[i], classes)
        return final[i]
    for name, indices in selections.items():
        if not indices or any(i < 0 or i >= len(candidates) for i in indices):
            raise ValueError('selection has invalid candidates')
        i = indices[int(np.argmax(values[indices]) if classes is not None else np.argmin(values[indices]))]
        outcomes[name] = dict(candidate=i, prediction=predict(i), clipped=False)
        if bounds is not None:
            outcomes[name + '_clipped'] = dict(candidate=i, prediction=np.clip(predict(i), *bounds), clipped=True)
            j = indices[int(np.argmin(bounded[indices]))]
            outcomes[name + '_bounded_selection'] = dict(candidate=j, prediction=np.clip(predict(j), *bounds), clipped=True)
    return outcomes, inner

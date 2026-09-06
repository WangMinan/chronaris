"""Deterministic balanced subject/record draws indexed by the saved data cursor."""
import hashlib
import json

from chronaris.modeling.training.candidate_validation import _batch_ids, _load_batch


class BalancedSampleSchedule:
    def __init__(self, sample_ids, hierarchy):
        if set(hierarchy) != set(sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sampling hierarchy must cover exactly the training samples")
        depths = {len(hierarchy[value]) for value in sample_ids}
        if len(depths) != 1 or not depths or min(depths) < 1:
            raise ValueError("sampling hierarchy depths must match and be nonempty")
        tree = {}
        for sample_id in sample_ids:
            node = tree
            for group in hierarchy[sample_id]:
                if not isinstance(group, str) or not group:
                    raise ValueError("sampling groups must be nonempty identifiers")
                node = node.setdefault(group, {})
            node[sample_id] = None
        self.tree = _freeze(tree)
        self.sha256 = hashlib.sha256(json.dumps(self.tree).encode()).hexdigest()

    def draw(self, start, count):
        if start < 0 or count <= 0:
            raise ValueError("sampling cursor/count is invalid")
        result = []
        for position in range(start, start + count):
            node = self.tree
            while isinstance(node[0], tuple):
                node, position = node[position % len(node)], position // len(node)
            result.append(node[position % len(node)])
        if len(set(result)) != len(result):
            raise ValueError("balanced actual batch repeats a sample; insufficient distinct records in a group")
        return tuple(result)


def _freeze(tree):
    if all(value is None for value in tree.values()):
        return tuple(sorted(tree))
    return tuple(_freeze(tree[key]) for key in sorted(tree))


def training_sample_schedule(batch, provider, sample_ids, batch_size, hierarchy=None):
    if hierarchy is None:
        hierarchy = {}
        for ids in _batch_ids(sample_ids, batch_size):
            loaded = _load_batch(batch, provider, ids)
            hierarchy.update({sample: (group,) for sample, group in zip(loaded.sample_ids, loaded.group_ids, strict=True)})
    return BalancedSampleSchedule(sample_ids, hierarchy)

"""Reader-facing names, kept separate from immutable source identifiers."""


def display_text(value):
    text = str(value)
    for old, new in (("鼎新", "实测数据集"), ("Dingxin", "实测数据集"), ("dingxin", "实测数据集"),
                     ("Chronaris", "融合系统"), ("chronaris", "融合系统"), ("Zorathos", "依托平台"),
                     ("离线案例", "观测记录"), ("内部验证片段", "观测片段"), ("冻结自监督模型", "自监督模型")):
        text = text.replace(old, new)
    return text


def present_result(result):
    """Adapt historical display strings without rewriting frozen result files."""
    result = dict(result)
    result["case"] = dict(result["case"])
    for key in ("title", "description", "model_label", "source_label"):
        result["case"][key] = display_text(result["case"].get(key, ""))
    return result

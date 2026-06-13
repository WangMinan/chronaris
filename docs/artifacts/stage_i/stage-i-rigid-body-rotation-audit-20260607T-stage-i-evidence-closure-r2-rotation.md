# Stage I Rigid-Body Rotation Audit - 20260607T-stage-i-evidence-closure-r2-rotation

- evidence_layer: `rotation_diagnostics`
- source_sortie_id: `20251005_四01_ACT-4_云_J20_22#01`
- vehicle_field_metadata: `{'status': 'loaded', 'field_count': 96, 'measurement': 'BUS6000019110020', 'analysis_id': 6000019110020, 'access_rule_id': 6000019510066, 'error': None}`
- rotation_status: `disabled`

## Stage H Feature Matches

| group | matched_fields |
| --- | --- |
| `speed` | `BUS6000019110020.code1027::[TSPI数据][载机平台系速度][速度_北向][_速度], BUS6000019110020.code1028::[TSPI数据][载机平台系速度][速度_西向][_速度]` |
| `acceleration` | `BUS6000019110020.code1024::[TSPI数据][载机平台系加速度][加速度_北向][_加速度], BUS6000019110020.code1025::[TSPI数据][载机平台系加速度][加速度_西向][_加速度], BUS6000019110020.code1036::[TSPI数据][载机法向过载][_过载]` |
| `altitude` | `BUS6000019110020.code1033::[TSPI数据][载机海拔高度][_高度], BUS6000019110020.code1043::[TSPI数据][载机气压高度][_高度], BUS6000019110020.code1044::[TSPI数据][载机卫星高度][_高度]ICD_1` |
| `vertical_speed` | `BUS6000019110020.code1026::[TSPI数据][载机平台系加速度][加速度_天向][_加速度], BUS6000019110020.code1029::[TSPI数据][载机平台系速度][速度_天向][_速度]` |
| `pitch` | `BUS6000019110020.code1030::[TSPI数据][载机俯仰角][_角度_毫弧度]` |
| `pitch_rate` | `-` |
| `roll` | `BUS6000019110020.code1032::[TSPI数据][载机横滚角][_角度_毫弧度]` |
| `roll_rate` | `-` |
| `yaw` | `BUS6000019110020.code1031::[TSPI数据][载机真航向][_角度_毫弧度]` |
| `yaw_rate` | `-` |

## MySQL Metadata Candidates

| group | matched_fields |
| --- | --- |
| `pitch` | `code1006::[TSPI数据][目标TSPI数据_有效性][有效性_俯仰角], BUS6000019110020.code1006::[TSPI数据][目标TSPI数据_有效性][有效性_俯仰角], code1030::[TSPI数据][载机俯仰角][_角度_毫弧度], BUS6000019110020.code1030::[TSPI数据][载机俯仰角][_角度_毫弧度]` |
| `pitch_rate` | `-` |
| `roll` | `code1007::[TSPI数据][目标TSPI数据_有效性][有效性_横滚角], BUS6000019110020.code1007::[TSPI数据][目标TSPI数据_有效性][有效性_横滚角], code1032::[TSPI数据][载机横滚角][_角度_毫弧度], BUS6000019110020.code1032::[TSPI数据][载机横滚角][_角度_毫弧度]` |
| `roll_rate` | `-` |
| `yaw` | `code1008::[TSPI数据][目标TSPI数据_有效性][有效性_真航向], BUS6000019110020.code1008::[TSPI数据][目标TSPI数据_有效性][有效性_真航向], code1031::[TSPI数据][载机真航向][_角度_毫弧度], BUS6000019110020.code1031::[TSPI数据][载机真航向][_角度_毫弧度]` |
| `yaw_rate` | `-` |

## Reading

1. 本审计只判断 `rotation` 是否具备真实启用条件，不把缺失字段包装成已验证物理约束。
2. 当前结论：`current sortie still lacks paired rate fields for pitch/roll/yaw`。
3. 当前 Stage H 缺口：`{'rotation': ['pitch/pitch_rate', 'roll/roll_rate', 'yaw/yaw_rate']}`。

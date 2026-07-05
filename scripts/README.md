# Scripts

运行约定：

- 默认使用 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。
- 脚本只承载 CLI 编排；可复用逻辑必须进入 `src/chronaris`。
- 长任务必须保留 `run.log`、`progress.json` 或等价进度记录。

## 当前目录

- `feature_export/`：标准化融合特征导出。
- `modeling/`：backbone 与 multitask training。
- `evaluation/dingxin/`：鼎新真实数据弱监督任务、组件消融、第三方对比和任务头校准。
- `evaluation/public_datasets/`：公开 UAB/NASA 数据适配、公开模型对比、公开融合校准和公开消融。
- `evidence/`：证据闭环、跨证据矩阵、指标校准、论文协议快照和图表材料。
- `runtime/`：runtime replay、inference 和 schema contract。
- `llm_preprocessing/`：LLM preprocessing 与对比。
- `archive/legacy_public_benchmark/`：历史公开 benchmark。

新增脚本必须进入上述职责目录，不要恢复阶段编号式脚本目录。

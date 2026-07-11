# MiniRocket 高维逻辑回归求解器运行时基准

状态：completed；本基准只使用 G1 train/validation，不读取 G2 锁定测试指标。

输入固定为 seed 17 生理单流的 384 个训练、96 个验证融合序列。已拟合的 10,000-kernel MiniRocket 将训练集变换为 `[384,9996]`，训练与验证变换合计 0.620 秒。默认多项 `lbfgs` 三值 C 网格所在完整 MiniRocket 组件约耗时 1743.68 秒；在同一冻结变换上，显式 one-vs-rest `liblinear` 的单个 C 拟合为 2.20–2.40 秒，三个值合计 6.97 秒。

求解器只按运行时与问题形态锁定，不按验证分数选择：约 10,000 维、小样本的 MiniRocket 分类头统一使用 `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`；64 维线性探针继续使用 `lbfgs`。所有方法、随机种子和任务共用同一规则与 C 网格，Chronaris 不获得额外搜索预算。

旧 `lbfgs` 模型保存在被忽略的 `artifacts/application_evaluation/2026-07-12_simulation-locked-consumers-lbfgs-runtime-benchmark/`，不会进入正式主表或论文结果。

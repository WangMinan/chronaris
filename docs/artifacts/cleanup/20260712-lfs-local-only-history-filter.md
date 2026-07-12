# Git LFS 本地留存与历史清理记录（2026-07-12）

## 结论

本轮将可复现的密集训练与运行产物改为开发机本地留存，并从 `main` 历史中删除；紧凑指标、协议清单、正式报告和论文图表继续由普通 Git 管理。处理目标是避免后续推送继续消耗 GitHub LFS 存储与下载流量，同时不删除开发机上的原始产物。

## 清理前基线

- `main` 历史包含 686 个唯一 Git LFS 对象，指针声明总量为 166,612,686 字节。
- 当前工作树命中 48 个本地留存文件，总量为 95,083,634 字节。
- 主要对象类型为训练曲线、逐批 GPU 性能记录、逐样本任务清单、预测明细、特征/序列包和相似度分布。

## 本地留存范围

以下路径模式由 `.gitignore` 管理，并从 `main` 历史中删除：

```text
docs/artifacts/**/training_curves*.csv
docs/artifacts/**/gpu_perf_batches*.csv
docs/artifacts/**/*task_manifest*.jsonl
docs/artifacts/**/raw_window_summary.jsonl
docs/artifacts/**/*predictions*.csv
docs/artifacts/**/t3_similarity_distribution.csv
docs/artifacts/**/llm_request_response_audit.jsonl
docs/artifacts/**/feature_bundle.npz
docs/artifacts/**/sequence_bundle.npz
docs/artifacts/**/*.partial.csv
```

上述文件仍保留在原工作树路径。额外恢复副本位于：

```text
artifacts/local_history_backup/2026-07-12_lfs-prune/
```

该目录同时保存清理前 `main` 的 Git bundle、LFS 对象库副本和文件哈希清单；仓库内另保留本地备份分支 `backup/main-before-lfs-filter-20260712`。

## 保留在 Git 中的证据

未列入上述模式的紧凑结果表、折级指标、实验注册表、表示清单、配置、协议、报告和图表继续保留。其历史中的 LFS 指针转换为普通 Git blob，不再触发 Git LFS 上传。

## 验收判据

1. 原路径下的 48 个本地文件与备份哈希一致。
2. `main` 不再跟踪本地留存模式。
3. `main` 历史不再包含 Git LFS 指针。
4. `git lfs status` 不再报告待上传对象。
5. 除本轮明确清理范围外，清理前后仓库树内容保持一致。

## 远端边界

历史重写能够阻止这些对象继续随 `main` 推送，但 GitHub 端已经上传的 LFS 对象可能继续计入存储。远端物理删除需要按 GitHub 的 LFS 清理流程另行处理；本轮不删除或重建远端仓库。

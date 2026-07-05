# Tests

运行测试时默认显式使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

当前测试按职责组织：

- `feature_export/`
- `modeling/`
- `evaluation/dingxin/`
- `evaluation/public_datasets/`
- `evidence/`
- `runtime/`
- `llm_preprocessing/`

全量验证命令：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q
```

若全量测试因 DB、CUDA 或本地数据依赖失败，记录失败原因，再运行不依赖外部条件的 focused tests。

"""Train the bounded Chapter 6 application model in an independent output directory."""
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == '__main__':
    import argparse
    import torch
    from chronaris.evaluation.application_tasks.chapter6_model import train_application_model
    parser = argparse.ArgumentParser(description='实测应用模型补充训练（CPU）')
    for name in ('source', 'snapshot', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT / 'artifacts'):
        parser.error('输出必须位于当前工作区 artifacts 独立目录')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(train_application_model(**vars(args)))

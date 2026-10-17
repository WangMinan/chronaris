#!/usr/bin/env python3
"""Register an existing trained model using an established observation schema."""
from run_fusion_workbench import PROJECT


def main():
    import argparse
    from pathlib import Path
    from chronaris.serving.fusion_workbench.catalog import Catalog
    parser = argparse.ArgumentParser(description='登记已完成训练的融合模型')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--schema-record',required=True,help='字段合同一致的已登记观测记录标识')
    parser.add_argument('--data-root',type=Path,default=PROJECT/'artifacts/fusion-workbench')
    args=parser.parse_args()
    if not args.data_root.resolve().is_relative_to(PROJECT) or args.data_root.resolve()==PROJECT:
        parser.error('模型目录必须位于本工作区内')
    catalog=Catalog(args.data_root/'cases');catalog.refresh()
    if args.schema_record not in catalog.cases:
        parser.error('字段参考记录不存在')
    print(catalog.register_model(args.checkpoint,catalog.cases[args.schema_record].schema))


if __name__=='__main__':
    main()

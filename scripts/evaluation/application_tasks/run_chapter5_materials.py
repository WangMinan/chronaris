"""Audit/export completed main training, or evaluate its frozen native consumers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','export','predict'))
    parser.add_argument('--main-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit')
    parser.add_argument('--unit',type=int)
    parser.add_argument('--role',choices=('validation','held_out'),default='validation')
    parser.add_argument('--sample-limit',type=int,default=4)
    parser.add_argument('--core-freeze')
    args=parser.parse_args()
    if args.action=='audit':
        from chronaris.evaluation.application_tasks.chapter5_assets import audit_main_models
        result=audit_main_models(args.main_root,args.output)
    elif args.action=='export':
        if not args.audit:parser.error('export requires --audit')
        from chronaris.evaluation.application_tasks.chapter5_export import export_main_materials
        result=export_main_materials(args.main_root,args.output,args.audit)
    else:
        if args.unit is None:parser.error('predict requires --unit')
        from chronaris.evaluation.application_tasks.chapter5_native import evaluate_main_unit
        result=evaluate_main_unit(main_root=args.main_root,index=args.unit,output_root=args.output,
            role=args.role,sample_limit=args.sample_limit,gate_path=args.core_freeze)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','data_files','historical_failures')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

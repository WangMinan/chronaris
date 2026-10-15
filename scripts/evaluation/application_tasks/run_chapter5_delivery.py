"""Run the bounded writing delivery using an unchanged frozen training workspace."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker'))
    parser.add_argument('--root',required=True)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--subset-root')
    parser.add_argument('--policy')
    parser.add_argument('--plot-root')
    parser.add_argument('--stage')
    parser.add_argument('--result')
    args=parser.parse_args()
    # Keep the old global source fingerprint and model implementation intact.
    sys.path.insert(0,str(Path(args.workspace).resolve()/'src'))
    module_path=Path(__file__).resolve().parents[3]/'src/chronaris/evaluation/application_tasks/chapter5_delivery.py'
    spec=importlib.util.spec_from_file_location('frozen_source_delivery',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.action=='prepare':
        if not all((args.subset_root,args.policy,args.plot_root)):parser.error('prepare requires subset, policy and plot roots')
        result=module.prepare(args.root,args.subset_root,args.policy,Path(__file__).resolve(),args.plot_root)
    elif args.action=='run':result=module.run(args.root,Path(__file__).resolve(),args.workspace)
    else:
        if not args.stage or not args.result:parser.error('worker requires stage and result')
        result=module.worker(args.root,args.stage)
        if result.get('status')=='waiting_gpu':raise SystemExit(75)
        module.write(args.result,result)
    print(result.get('status',result.get('format')),flush=True)


if __name__=='__main__':main()

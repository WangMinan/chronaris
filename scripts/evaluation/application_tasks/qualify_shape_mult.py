"""Create separately qualified fold inputs after source-native extraction."""
import argparse
from chronaris.evaluation.application_tasks.shape_mult_inputs import qualify_shape_inputs

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('native','original','report','assets'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=qualify_shape_inputs(a.native,a.original,a.report,a.assets);print(r['status'],r['windows'])

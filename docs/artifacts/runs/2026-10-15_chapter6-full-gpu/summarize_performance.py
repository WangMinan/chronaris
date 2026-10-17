"""Recompute all device/scale summaries using only the saved CSV and standard library."""
import csv
from pathlib import Path
from statistics import mean,stdev
root=Path(__file__).parent
rows=list(csv.DictReader((root/'performance.csv').open()))
metrics=('model_load_s','normalized_s','input_transfer_s','forward_s','output_transfer_pool_s',
 'chunk_write_s','standard_export_readback_csv_s','result_json_zip_s','pipeline_total_s',
 'http_completed_s','http_result_s','http_delivery_s','windows_per_s','VmRSS_mib','VmHWM_mib',
 'gpu_peak_allocated_mib','gpu_peak_reserved_mib')
summary=[]
for device in ('cpu','cuda:0'):
 for n in (1,4,16):
  selected=[r for r in rows if r['device']==device and int(r['windows'])==n]
  assert len(selected)==3 and {int(r['repeat']) for r in selected}=={1,2,3}
  for metric in metrics:
   values=[float(r[metric]) for r in selected];assert all(v>=0 for v in values)
   summary.append(dict(device=device,windows=n,metric=metric,n=3,mean=mean(values),sample_std=stdev(values),minimum=min(values),maximum=max(values)))
with (root/'performance-summary.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
print('18 raw measurements summarized')

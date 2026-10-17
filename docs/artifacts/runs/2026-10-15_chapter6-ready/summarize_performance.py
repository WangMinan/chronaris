"""Recompute performance summaries from measured CSV; Python standard library only."""
import csv
from pathlib import Path
from statistics import mean, stdev

root = Path(__file__).parent
with (root / 'performance.csv').open() as handle:
    rows = list(csv.DictReader(handle))
metrics = ('model_load_s', 'input_read_s', 'inference_s', 'chunk_write_s',
           'standard_export_readback_csv_s', 'result_json_zip_s', 'pipeline_total_s',
           'http_completed_s', 'http_result_s', 'http_delivery_s', 'windows_per_s', 'VmHWM_mib')
summary = []
for count in sorted({int(row['windows']) for row in rows}):
    selected = [row for row in rows if int(row['windows']) == count]
    assert len(selected) == 3 and {int(row['repeat']) for row in selected} == {1, 2, 3}
    for metric in metrics:
        values = [float(row[metric]) for row in selected]
        assert all(value >= 0 for value in values)
        summary.append(dict(windows=count, metric=metric, n=len(values), mean=mean(values),
                            sample_std=stdev(values), minimum=min(values), maximum=max(values)))
with (root / 'performance-summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print('Wrote', root / 'performance-summary.csv')

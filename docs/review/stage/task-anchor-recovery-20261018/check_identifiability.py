"""Algebra checks for the proposed protocol, not a model-recovery benchmark."""
import json
from pathlib import Path
import numpy as np


def main():
    reference=np.linspace(0,180,3601)
    rows=[]
    for offset in (-3.,0.,3.):
        for drift in (-100.,0.,100.):
            a=1+drift*1e-6;observed=a*reference+offset
            corrected=(observed-offset)/a
            error=float(abs(corrected-reference).max());assert error<1e-12
            rows.append(dict(offset_s=offset,drift_ppm=drift,max_inverse_error_s=error,
                max_uncorrected_error_s=float(abs(observed-reference).max())))
    # Equal total lag gives exactly the same visible signals with different clock and response delays.
    sensor=np.linspace(40,140,2001)
    signal=lambda t:np.sin(.17*t)+.3*np.cos(.061*t)+np.exp(-((t-75)/5)**2)
    pairs=[(-1.,5.),(1.,3.),(3.,1.)]
    values=[signal(sensor-(offset+delay)) for offset,delay in pairs]
    differences=[float(abs(x-values[0]).max()) for x in values]
    assert max(differences)==0
    # Same physical response and an externally observed common pulse make clock offsets distinguishable.
    pulse_reference=np.array([20.,60.,100.,140.])
    pulse_times=[pulse_reference+offset for offset,_ in pairs]
    assert not np.array_equal(pulse_times[0],pulse_times[1])
    result=dict(status='passed',scope='closed-form protocol examples only; no fitted estimator, model inference, or training',
        affine_cases=rows,ambiguous_pairs=[dict(offset_s=b,response_delay_s=d,total_lag_s=b+d) for b,d in pairs],
        visible_signal_max_differences=differences,
        positive_offset_requires_subtraction=True,independent_sync_pulse_breaks_this_ambiguity=True,
        clock_drift_100ppm_over_30s_s=.003,clock_drift_100ppm_over_180s_s=.018)
    Path(__file__).with_name('identifiability_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('9 affine inverse checks and 3 indistinguishable signal pairs passed; no training')


if __name__=='__main__':main()

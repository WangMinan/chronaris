"""Native forecast of the strictly historical five-second mean channel."""
import torch


def load_forecast_model(assets,device='cuda'):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(assets['chronos2']['path'],device_map=device,
                                           torch_dtype=torch.float32,local_files_only=True).model


def forecast_point(model,values,mask,center,scale):
    if values.shape!=mask.shape or values.ndim!=3 or values.shape[1]!=96:raise ValueError('forecast input shape')
    config=model.chronos_config
    assert config.output_patch_size==16 and config.quantiles.count(.5)==1
    device=next(model.parameters()).device;b,length,c=values.shape
    x=values.transpose(1,2).to(device=device,dtype=torch.float32)
    valid=mask.transpose(1,2).to(device=device,dtype=torch.bool)
    output=model(context=x.reshape(b*c,length),context_mask=valid.reshape(b*c,length),
                 group_ids=torch.arange(b,device=device).repeat_interleave(c),num_output_patches=1)
    forecasts=output.quantile_preds.reshape(b,c,len(config.quantiles),16)
    prediction=forecasts[:,-1,config.quantiles.index(.5),15]*scale+center
    if not torch.isfinite(prediction).all():raise ValueError('nonfinite native forecast')
    return prediction

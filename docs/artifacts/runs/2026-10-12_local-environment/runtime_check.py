import importlib
import json
import math
import platform
from pathlib import Path
import torch
from torchdiffeq import odeint

out = Path('/home/wangminan/chronaris-migration-20261012')
modules = ['numpy','scipy','pandas','pyarrow','sklearn','numba','aeon','matplotlib','torch','torchdiffeq','transformers','chronos','chronaris']
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = {'version':str(getattr(module,'__version__','package')), 'path':module.__file__}
assert torch.cuda.is_available()
torch.set_num_threads(4)
with torch.inference_mode():
    torch.manual_seed(17)
    a = torch.randn(128,128,device='cuda')
    actual = a @ a.T
    torch.testing.assert_close(actual.cpu(), a.cpu() @ a.cpu().T, atol=1e-4, rtol=1e-4)
    compiled = torch.compile(lambda x: torch.sin(x) + x.square(), fullgraph=True)
    torch.testing.assert_close(compiled(a), torch.sin(a)+a.square())
    solved = odeint(lambda t,y: -y, torch.ones(2,device='cuda'), torch.tensor([0.,1.],device='cuda'))
    torch.testing.assert_close(solved[-1], torch.full((2,),math.exp(-1),device='cuda'),atol=1e-5,rtol=1e-5)
    torch.cuda.synchronize()
result = {'status':'passed','python':platform.python_version(),'modules':versions,'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,'compute_capability':torch.cuda.get_device_capability(),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'checks':['imports','cuda_matmul_vs_cpu','torch_compile_forward','cuda_ode_solver'],'training_updates':0}
(out/'runtime-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

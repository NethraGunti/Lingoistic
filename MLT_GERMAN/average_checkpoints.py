import torch
import os
from collections import OrderedDict

source_folder = "./"
starts_with = "step"
ends_with = ".pth.tar"

checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(starts_with) and f.endswith(ends_with)]
assert len(checkpoint_names) > 0, "Did not find any checkpoints!"
averaged_params = OrderedDict()
for c in checkpoint_names:
    checkpoint = torch.load(c)['model']
    checkpoint_params = checkpoint.state_dict()
    checkpoint_param_names = checkpoint_params.keys()
    for param_name in checkpoint_param_names:
        if param_name not in averaged_params:
            averaged_params[param_name] = checkpoint_params[param_name].clone() * 1 / len(checkpoint_names)
        else:
            averaged_params[param_name] += checkpoint_params[param_name] * 1 / len(checkpoint_names)
averaged_checkpoint = torch.load(checkpoint_names[0])['model']
for param_name in averaged_checkpoint.state_dict().keys():
    assert param_name in averaged_params
averaged_checkpoint.load_state_dict(averaged_params)
torch.save({'model': averaged_checkpoint}, "averaged_transformer_checkpoint.pth.tar")

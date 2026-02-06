import torch
import pathlib
import yaml
import force_estimation.model.CFModel as CFModel

with open("config.yaml", 'r+', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = CFModel(config['encoder'],
                     config['decoder'],
                     config['num_decoder_layer'],
                     config['dim_decoder_hidden'],
                     config['target_index'],
                     config['image_size'])

save_dir = pathlib.Path(config['model_dir']).parent.__str__()
checkpoint = torch.load(config['model_dir'])
model.load_state_dict(checkpoint)

for name, p in model.named_parameters():
    print(name, p.abs().max().item())

import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)
print("True" if torch.cuda.is_available() else "False")
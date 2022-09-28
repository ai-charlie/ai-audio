import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.__version__, torchaudio.__version__, torch.cuda.is_available(),device)
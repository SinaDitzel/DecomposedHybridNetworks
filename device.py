import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    is_cuda = True
else:
    device = torch.device("cpu")
    is_cuda = False
#device = torch.device( "cpu");
cpu = torch.device("cpu")
print("using", device)

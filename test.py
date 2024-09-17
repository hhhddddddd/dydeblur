import torch

A = torch.randn(1, 5, 5)  
B = torch.randn(1, 5, 5)  

C, _ = torch.max(torch.stack((A, B)), dim=0)

print(C.shape)  
print(A,B,C)
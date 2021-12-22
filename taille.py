import torch

x = torch.rand([256,1,28,28])

print(x.size())
x = torch.cat((x,x,x),1)

print (x.size())


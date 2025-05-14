import torch
im=torch.randn([10,10], requires_grad=True)
im=im.view([100])
w=torch.randn([100], requires_grad=True)

s=w@im
s.backward()

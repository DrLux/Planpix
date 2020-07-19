import torch


print("Using gpu: ", torch.cuda.current_device())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
  device = torch.device("cuda")          # a CUDA device object
  x = torch.tensor([1,2,3])
  y = torch.tensor([4,5,6]).cuda() #la y la sposto su gpu
  print(x)
  print(y)
  #print(x+y) non si può fare
else:
  print("no gpu available")

while True:
    a = 1
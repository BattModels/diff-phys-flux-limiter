import torch

import sys
sys.path.append('/home/chyhuang/research/flux_limiter/src')
from models.model import FluxLimiter, SymmetricFluxLimiter
from utils import fvm_solver, utils

torch.manual_seed(3407)

device = 'cpu'

# Model
model = FluxLimiter(1,1,64,5,act="relu")
# model = SymmetricFluxLimiter(1,1,64,5,act="tanh") #
model.load_state_dict(torch.load("model_linear_relu.pt", map_location=torch.device(device)))
model = model.to(device)

r = torch.tensor([1.0], requires_grad=True)
phi = model(r)
phi.backward()
print(r.grad)


# Set the print threshold to a very high value so that nothing is omitted
torch.set_printoptions(threshold=sys.maxsize)

# Open a text file in write mode
with open("model_linear_relu_weights.txt", "w") as file:
    for name, param in model.state_dict().items():
        file.write(f"{name}:\n")
        file.write(str(param))
        file.write("\n\n")
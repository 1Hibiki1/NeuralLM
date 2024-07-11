import torch
import torch.nn as nn
from kan import KANLinear

class VNN(nn.Module):
    def __init__(self, layer_config, device='cpu'):
        super(VNN, self).__init__()

        self.layers = list()
        for i in range(len(layer_config) - 1):
            weights = torch.Tensor(layer_config[i+1], layer_config[i])
            nn.init.xavier_uniform_(weights)
            self.layers.append(weights)

        self.layers = nn.ParameterList(self.layers)
        self.layers.to(device)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = torch.matmul(layer, output)
        output += inputs
        output = (output / torch.norm(output, p=2, dim=2, keepdim=True))

        return output
    
class NeuralLMBlock(nn.Module):
    def __init__(self, layer_config, dim, device='cpu'):
        super(NeuralLMBlock, self).__init__()
        self.vnn = VNN(layer_config, device=device)
        up_proj_factor = 1.5
        self.up_proj = nn.Linear(dim, int(dim*up_proj_factor))
        self.down_proj = nn.Linear(int(dim*up_proj_factor), dim)
    
    def forward(self, inputs):
        output = inputs
        output = self.vnn(output)
        output = self.up_proj(output)
        output = self.down_proj(output)
        return output
    
class NeuralLMEncoder(nn.Module):
    def __init__(self, layer_config, dim, n_blocks, device='cpu'):
        super(NeuralLMEncoder, self).__init__()
        self.blocks = nn.ModuleList([NeuralLMBlock(layer_config, dim, device=device) for _ in range(n_blocks)])
    
    def forward(self, inputs):
        output = inputs
        for block in self.blocks:
            output = block(output)
        return output

if __name__ == "__main__":
    inputs = torch.ones(4, 3).to('mps')
    n = VNN([4, 3])
    print("\nOUTPUT:")
    print(n.forward(inputs))

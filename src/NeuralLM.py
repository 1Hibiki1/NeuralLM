import torch
import torch.nn as nn
from kan import KANLinear


class VNN(nn.Module):
    def __init__(self, layer_config):
        super(VNN, self).__init__()

        self.layers = list()
        for i in range(len(layer_config) - 1):
            weights = torch.Tensor(layer_config[i+1], layer_config[i])
            nn.init.xavier_normal_(weights)
            self.layers.append(weights)

        self.layers = nn.ParameterList(self.layers)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = torch.matmul(layer, output)
        return output


class NeuralLMBlock(nn.Module):
    def __init__(self, layer_config, dim):
        super(NeuralLMBlock, self).__init__()
        self.vnn = VNN(layer_config)
        up_proj_factor = 1.5
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim*up_proj_factor)),
            nn.GELU(),
            nn.Linear(int(dim*up_proj_factor), dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, inputs):
        output = inputs
        output = output + self.ln1(self.vnn(output))
        output = output + self.ln2(self.ffn(output))
        return output
    
    def norm(self, inputs):
        return (inputs / torch.norm(inputs, p=2, dim=2, keepdim=True))


class NeuralLMEncoder(nn.Module):
    def __init__(self, layer_config, dim, n_blocks):
        super(NeuralLMEncoder, self).__init__()
        self.blocks = nn.ModuleList(
            [NeuralLMBlock(layer_config, dim) for _ in range(n_blocks)]
        )

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

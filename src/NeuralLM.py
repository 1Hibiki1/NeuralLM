import torch
import torch.nn as nn
from kan import KANLinear
from torchtune.modules import RMSNorm


class NeuralLMConfig:
    def __init__(
        self,
        embedding_dim = 768,
        n_input = 512,
        layer_config = [512],
        n_blocks = 1,
        ffn_up_proj_factor = 1.5
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_input = n_input
        self.layer_config = [n_input] + layer_config
        self.n_blocks = n_blocks
        self.ffn_up_proj_factor = ffn_up_proj_factor


class VNN(nn.Module):
    def __init__(self, layer_config):
        super(VNN, self).__init__()

        self.layers = list()
        for i in range(len(layer_config) - 1):
            weights = torch.Tensor(layer_config[i+1], layer_config[i])
            nn.init.xavier_uniform_(weights)
            self.layers.append(weights)

        self.layers = nn.ParameterList(self.layers)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = torch.matmul(layer, output)
        return output
    
    
class NeuralLMFFN(nn.Module):
    def __init__(self, config: NeuralLMConfig):
        super(NeuralLMFFN, self).__init__()
        self.up_proj = nn.Linear(config.embedding_dim, int(config.embedding_dim*config.ffn_up_proj_factor), bias=False)
        # self.up_proj = KANLinear(config.embedding_dim, int(config.embedding_dim*config.ffn_up_proj_factor))
        nn.init.xavier_uniform_(self.up_proj, gain=nn.init.calculate_gain('relu'))

        self.gelu = nn.GELU()

        self.down_proj = nn.Linear(int(config.embedding_dim*config.ffn_up_proj_factor), config.embedding_dim, bias=False)
        # self.down_proj = KANLinear(int(config.embedding_dim*config.ffn_up_proj_factor), config.embedding_dim)
        nn.init.xavier_uniform_(self.down_proj)

    def forward(self, inputs):
        output = inputs
        output = self.up_proj(output)
        output = self.gelu(output)
        output = self.down_proj(output)
        return output


class NeuralLMBlock(nn.Module):
    def __init__(self, config: NeuralLMConfig):
        super(NeuralLMBlock, self).__init__()
        self.vnn = VNN(config.layer_config)
        
        self.ln1 = RMSNorm(config.embedding_dim)
        self.ffn = NeuralLMFFN(config)
        self.ln2 = RMSNorm(config.embedding_dim)

    def forward(self, inputs):
        output = inputs
        output = output + self.vnn(self.ln1(output))
        output = output + self.ffn(self.ln2(output))
        return output


class NeuralLMEncoder(nn.Module):
    def __init__(self, config: NeuralLMConfig):
        super(NeuralLMEncoder, self).__init__()
        self.blocks = nn.ModuleList(
            [NeuralLMBlock(config) for _ in range(config.n_blocks)]
        )

    def forward(self, inputs):
        output = inputs
        for block in self.blocks:
            output = block(output)
        return output

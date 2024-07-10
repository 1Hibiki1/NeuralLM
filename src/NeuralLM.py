import torch
import torch.nn as nn


class VNN(nn.Module):
    def __init__(self, layer_config, device='cpu'):
        super(VNN, self).__init__()

        self.layers = list()
        for i in range(len(layer_config) - 1):
            weights = torch.Tensor(layer_config[i+1], layer_config[i])
            nn.init.xavier_uniform_(weights)
            self.layers.append(weights.to(device))

        self.layers = nn.ParameterList(self.layers)

        # transformers likes to use GPU, will have to change this to cuda once we switch platforms
        self.layers.to(device)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = torch.matmul(layer, output)
        output += inputs
        output = (output / torch.norm(output, p=2, dim=1, keepdim=True))

        return output


if __name__ == "__main__":
    inputs = torch.ones(4, 3).to('mps')
    n = VNN([4, 3])
    print("\nOUTPUT:")
    print(n.forward(inputs))

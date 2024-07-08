import torch
import torch.nn as nn


class VectorNeuron(nn.Module):
    def __init__(self, input_dim, num_inputs):
        super(VectorNeuron, self).__init__()

        # Learnable weights
        # self.weights = nn.Parameter(torch.ones(num_inputs))
        self.weights = nn.Parameter(torch.Tensor(num_inputs, input_dim))
        nn.init.xavier_uniform_(self.weights)

        self.input_dim = input_dim
        self.num_inputs = num_inputs

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs, f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        weighted_sum = sum(self.weights[i] * inputs[i]
                           for i in range(self.num_inputs))
        return weighted_sum


class VectorNeuronLayer(nn.Module):
    def __init__(self, input_dim, num_inputs, num_neurons):
        super(VectorNeuronLayer, self).__init__()
        self.neurons = nn.ModuleList(
            [VectorNeuron(input_dim, num_inputs) for _ in range(num_neurons)])

    def forward(self, *inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return torch.stack(outputs, dim=0)  # Stack the outputs of all neurons


class VectorNeuralNetwork(nn.Module):
    def __init__(self, layer_config, input_dim):
        super(VectorNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        num_inputs = layer_config[0]

        for num_neurons in layer_config[1:]:
            layer = VectorNeuronLayer(input_dim, num_inputs, num_neurons)
            self.layers.append(layer)
            # The output of the current layer becomes the input to the next
            num_inputs = num_neurons

    def forward(self, *inputs):
        x = inputs
        for layer in self.layers:
            x = layer(*x)  # Unpack the inputs for the next layer
            # Unbind the tensor back to a tuple of tensors for the next layer
            x = x.unbind(0)
        return torch.stack(x, dim=0)  # Stack the final output
    
    
class VNN(nn.Module):
    def __init__(self, layer_config):
        super(VNN, self).__init__()
        
        self.layers = list()
        for i in range(len(layer_config) - 1):
            weights = nn.Parameter(torch.Tensor(layer_config[i+1], layer_config[i]))
            nn.init.xavier_uniform_(weights)
            self.layers.append(weights)
    
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = torch.matmul(layer, output)
        return output


if __name__ == "__main__":
    input_dim = 3  # Dimension of each vector x

    print('-'*80)
    print("Single Neuron")

    num_inputs = 3
    inputs = torch.ones(num_inputs, input_dim)
    neuron = VectorNeuron(input_dim, num_inputs)

    print(f"input dim: {input_dim}, num inputs: {num_inputs}")

    print("\nInputs:")
    print(inputs)
    print("\nOutput")
    print(neuron(inputs))
    print('-'*80)

    print('-'*80)
    print("Single Layer")

    num_inputs = 3
    num_neurons = 2
    layer = VectorNeuronLayer(input_dim, num_inputs, num_neurons)
    inputs = torch.ones(num_inputs, input_dim)

    print(f"input dim: {input_dim}, num inputs: {
        num_inputs}, num neurons: {num_neurons}")

    print("\nInputs:")
    print(inputs)
    print("\nOutput")
    print(layer(*inputs))
    print('-'*80)

    print('-'*80)
    print("Multiple Layers")

    num_inputs = 3
    layer_config = [num_inputs, 2, 1]
    network = VectorNeuralNetwork(layer_config, input_dim)
    inputs = torch.ones(num_inputs, input_dim)

    print(f"input dim: {input_dim}, config: {layer_config}")

    print("\nInputs:")
    print(inputs)
    print("\nOutput")
    print(network(*inputs))
    print('-'*80)

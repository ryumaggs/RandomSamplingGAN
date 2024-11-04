import torch
import numpy as np
import random
def dict2vector(input):
    if type(input) == dict:
        vectors = []
        for key, value in input.items():
            if type(value) == list:
                vectors += value
            else:
                vectors.append(value)
    else:
        vectors = []
        for dict_element in input:
            vector = []
            for key, value in dict_element.items():
                if type(value) == list:
                    vector += value
                else:
                    vector.append(value)
            vectors.append(vector)

    return vectors

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def to_onehot(y, num_classes=5):
    return np.eye(num_classes)[y].reshape(-1, num_classes)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
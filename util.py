import torch
import numpy as np
import random

def compute_data_shape(embedding_dict,X_tensor):
    '''
    input - embedding dict - dictinoary indicating which feature indexes should be embedded
    X_tensor - torch tensor - data set
    '''
    final_shape = 0
    for feat_idx in range(X_tensor.shape[1]):
        if feat_idx in embedding_dict:
            final_shape += embedding_dict[feat_idx][2]
        else:
            # keep raw feature (as float)
            final_shape += 1

    return final_shape

def create_embedding_layers(embedding_dict,device):
    embedding_layers = {}
    for feat_idx in embedding_dict:
        method = embedding_dict[feat_idx][0]
        original_size = embedding_dict[feat_idx][1]
        new_size = embedding_dict[feat_idx][2]
        if method == "embed":
            embedding_layers[feat_idx] = torch.nn.Embedding(original_size, new_size).to(device)
    return embedding_layers

def embed_data(embedding_layers,embedding_dict,X_tensor,overrite_start_idx=0):
    '''
    requires input to be torch tensor
    '''
    encoded_features = []
    for feat_idx in range(X_tensor.shape[1]):
        if feat_idx in embedding_dict:
            nfi = feat_idx + overrite_start_idx
            method = embedding_dict[nfi][0]
            original_size = embedding_dict[nfi][1]
            new_size = embedding_dict[nfi][2]
            if method == "embed":
                out = embedding_layers[nfi](X_tensor[:, feat_idx].long())
            elif method == "onehot":
                out = torch.nn.functional.one_hot(X_tensor[:, feat_idx].long(), num_classes=original_size).float()
            encoded_features.append(out)
        else:
            # keep raw feature (as float)
            encoded_features.append(X_tensor[:, feat_idx].float().unsqueeze(1))
    final_tensor = torch.cat(encoded_features, dim=1)
    return final_tensor

def jsd_from_samples(x, y, base=2):
    """
    Jensen–Shannon divergence between two sample arrays of a categorical variable.
    
    Parameters
    ----------
    x, y : 1D np.ndarray
        Raw categorical samples (ints, strings, or any hashable values).
    base : int
        Logarithm base (2 -> divergence in [0,1], natural log -> nats).

    Returns
    -------
    float
        Jensen–Shannon divergence.
    """
    # Ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Get the union of categories
    categories = np.unique(np.concatenate([x, y]))

    # Build probability vectors
    Px = np.array([np.mean(x == c) for c in categories])
    Qy = np.array([np.mean(y == c) for c in categories])

    # Avoid log(0)
    eps = 1e-12
    Px = np.clip(Px, eps, 1.0)
    Qy = np.clip(Qy, eps, 1.0)
    Px /= Px.sum()
    Qy /= Qy.sum()

    # Mixture distribution
    M = 0.5 * (Px + Qy)

    # KL divergence helper
    def kl(A, B):
        return np.sum(A * (np.log(A) - np.log(B)))

    val = 0.5 * kl(Px, M) + 0.5 * kl(Qy, M)
    if base == 2:
        val /= np.log(2)
    return val

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

def set_seed_old(seed, device):
    print('SETTING SEED: ', seed)
    rngs = {}
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)
    rngs['torch'] = torch_rng
    torch_cuda_rng = torch.Generator(device=device)
    torch_cuda_rng.manual_seed(seed)
    rngs['torch_cuda'] = torch_cuda_rng
    python_rng = random.Random(seed)
    rngs['python'] = python_rng
    np_rng = np.random.default_rng(seed)
    rngs['np'] = np_rng
    rngs['seed'] = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return rngs

def set_seed(seed,
             device,
              data_init=[True, True],
              data_gen =True,
              network_init=True,):
    rngs = {}
    #torch_cuda_rng = torch.Generator(device=device)
    #torch_cuda_rng.manual_seed(seed)
    #rngs['torch_cuda'] = torch_cuda_rng
    #python_rng = random.Random(seed)
    #rngs['python'] = python_rng

    #this generator is only used in the initial network initializations:
    if network_init:
        torch_rng = torch.Generator()
        torch_rng.manual_seed(seed)
        rngs['torch'] = torch_rng
    else:
        torch_rng = torch.Generator()
        torch_rng.manual_seed(np.random.randint(1e6))
        rngs['torch'] = torch_rng
        

    #these generators is only used when sampling GT and Bias samples during training
    #np_rng picks the GT and Bias data set 
    #torch_cuda is used by Gumbel softmax as well as Compute_gradient_penalty
    if data_gen is True:
        np_rng = np.random.default_rng(seed)
        rngs['np'] = np_rng
        cuda_rng = torch.Generator(device=device)
        cuda_rng.manual_seed(seed)
        rngs['torch_cuda'] = cuda_rng
    else:
        np_rng = np.random.default_rng(np.random.randint(1e6))
        rngs['np'] = np_rng
        cuda_rng = torch.Generator(device=device)
        cuda_rng.manual_seed(np.random.randint(1e6))
        rngs['torch_cuda'] = cuda_rng

    #this generator is only used when first choosing GT and bias samples for training
    same_gt = data_init[0]
    same_bias = data_init[1]
    if same_gt:
        rngs['seed_gt'] = seed
    else:
        rngs['seed_gt'] = np.random.randint(1e6)
    if same_bias:
        rngs['seed_bias'] = seed
    else:
        rngs['seed_bias'] = np.random.randint(1e6)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return rngs

def get_avg_grad_per_layer(model):
    grad_sum = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            avg_grad = param.grad.abs().mean().item()
            grad_sum += avg_grad

    return grad_sum

def normalize_to_minus1_plus1(data1,data2):
    """
    Normalize a 2D tensor dataset (samples x features) feature-wise to [-1, 1].

    Args:
        data (torch.Tensor): shape (N, D), float or double

    Returns:
        normalized_data (torch.Tensor): same shape, normalized to [-1, 1]
        data_min (torch.Tensor): shape (D,), min of each feature before scaling
        data_max (torch.Tensor): shape (D,), max of each feature before scaling
    """
    data_min = np.array([min(x,y) for x,y in zip(np.min(data1, axis=0), np.min(data2, axis=0))])
    data_max = np.array([max(x,y) for x,y in zip(np.max(data1, axis=0), np.max(data2, axis=0))])

    # Avoid division by zero for constant features
    denom = data_max - data_min
    denom[denom == 0] = 1.0

    # Scale to [0,1]
    data_norm_0_1 = (data1 - data_min) / denom
    # Scale to [-1, 1]
    data_norm_1 = data_norm_0_1 * 2 - 1

    # Scale to [0,1]
    data_norm_0_2 = (data2 - data_min) / denom
    # Scale to [-1, 1]
    data_norm_2 = data_norm_0_2 * 2 - 1
    return data_norm_1, data_norm_2

def warmup_spectral_norm(model, input_shape, device=torch.device('cuda:0'), steps=5):
    print("in warmup")
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    for _ in range(steps):
        _ = model(dummy_input)
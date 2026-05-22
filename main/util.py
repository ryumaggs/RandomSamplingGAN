import torch
import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def load_config(path=None):
    _DEFAULT_CONFIG_PATH = "./configs/default_config.yaml"
    """Load a YAML config. Falls back to default_config.yaml if no path given."""
    with open(path or _DEFAULT_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
def rename_check(df, old_name, new_name=None):
    if new_name is not None:
        df.rename(columns={old_name: new_name}, inplace=True)

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
    if not isinstance(X_tensor, torch.Tensor):
        X_local = torch.tensor(X_tensor)
    else:
        X_local = X_tensor
    encoded_features = []
    for feat_idx in range(X_local.shape[1]):
        if feat_idx in embedding_dict:
            nfi = feat_idx + overrite_start_idx
            method = embedding_dict[nfi][0]
            original_size = embedding_dict[nfi][1]
            new_size = embedding_dict[nfi][2]
            if method == "embed":
                out = embedding_layers[nfi](X_local[:, feat_idx].long())
            elif method == "onehot":
                out = torch.nn.functional.one_hot(X_local[:, feat_idx].long(), num_classes=original_size).float()
            encoded_features.append(out)
        else:
            # keep raw feature (as float)
            encoded_features.append(X_local[:, feat_idx].float().unsqueeze(1))

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

def plot_heatmaps(matrices, labels_dict=None, titles=None):
    """
    Plots a list of square matrices as heatmaps with properly aligned top labels.
    """
    cmap = matplotlib.cm.get_cmap("seismic").copy()
    cmap.set_bad(color="white")

    n = len(matrices)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    # Global symmetric color limits
    all_vals = np.concatenate([M.flatten() for M in matrices])
    vmax = max(abs(all_vals.min()), abs(all_vals.max()))
    vmin = -vmax

    for i, M in enumerate(matrices):
        size = M.shape[0]
        M_masked = np.ma.array(M, mask=np.eye(size, dtype=bool))
        ax = axes[i]

        im = ax.imshow(M_masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        # Labels
        if labels_dict and i in labels_dict:
            labels = labels_dict[i]
        else:
            labels = [str(j) for j in range(size)]

        # Explicit tick positions
        positions = np.arange(size)

        # X-axis (bottom)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

        # X-axis (top)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(positions)
        ax_top.set_xticklabels(labels, rotation=45, ha='center', fontsize=8)
        ax_top.xaxis.set_ticks_position('top')
        ax_top.xaxis.set_label_position('top')

        # Y-axis
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)

        # Masked diagonal
        ax.imshow(M_masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

        if titles:
            ax.set_title(titles[i])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(len(matrices), len(axes)):
        axes[j].axis('off')

    plt.show()

def plot_heatmaps_multiplicative(matrices, labels_dict=None, titles=None):
    """
    Plots multiplicative-scale heatmaps using SymLogNorm for better visibility.
    Input matrices are raw logit differences.
    """
    # Convert logits → multiplicative
    mult_matrices = [np.exp(M) for M in matrices]

    cmap = matplotlib.cm.get_cmap("seismic").copy()
    cmap.set_bad(color="white")

    n = len(mult_matrices)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    # Global vmin/vmax
    all_vals = np.concatenate([M.flatten() for M in mult_matrices])
    vmin = all_vals.min()
    vmax = all_vals.max()

    # Symmetric around 1
    # linthresh controls sensitivity near 1 (5% recommended)
    norm = matplotlib.colors.SymLogNorm(
        linthresh=0.05,  # ±5% region is linear
        vmin=vmin,
        vmax=vmax,
        base=10
    )

    for i, M in enumerate(mult_matrices):
        size = M.shape[0]

        # Mask the diagonal
        M_masked = np.ma.array(M, mask=np.eye(size, dtype=bool))

        ax = axes[i]
        im = ax.imshow(M_masked, cmap=cmap, norm=norm, interpolation='nearest')

        # Labels
        if labels_dict and i in labels_dict:
            labels = labels_dict[i]
        else:
            labels = [str(j) for j in range(size)]

        positions = np.arange(size)

        # Bottom x-axis
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

        # Top x-axis
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(positions)
        ax_top.set_xticklabels(labels, rotation=45, ha='center', fontsize=8)
        ax_top.xaxis.set_ticks_position('top')

        # Y-axis
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)

        if titles:
            ax.set_title(titles[i])

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for j in range(len(mult_matrices), len(axes)):
        axes[j].axis('off')

    plt.show()
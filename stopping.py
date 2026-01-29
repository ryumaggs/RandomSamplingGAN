import pickle
import numpy as np
import torch

from Discriminator import *
from util import set_seed
from typing import Dict, Hashable, Tuple, Optional
from tqdm import tqdm

def pairwise_weighted_minibatch_scores(
    A: Dict[int, np.ndarray],
    B: Dict[int, nn.Module],
    X: np.ndarray,
    K: int,
    minibatch_size: int = 32,
    *,
    replace: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    use_eval: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pairing (a_key in A, b_key in B):
      - sample K minibatches (each of size minibatch_size) from X using weights A[a_key]
      - stack them into a single tensor of shape (K, minibatch_size, D)
      - run network B[b_key] on that tensor
      - reduce the output to a scalar (mean over all elements)
      - store the scalar in a numpy matrix: result[i, j] corresponds to (A_keys[i], B_keys[j])

    Parameters
    ----------
    A : dict[int, np.ndarray]
        Keys are ints; values are nonnegative weights over rows of X with shape (M,).
        They do not need to be normalized (we will normalize internally).
    B : dict[int, torch.nn.Module]
        Keys are ints; values are neural nets that accept input shape (batch=K, mini-batch, D).
    X : np.ndarray
        Dataset of shape (M, D).
    K : int
        Number of randomly sampled minibatches per A key.
    minibatch_size : int
        Size of each minibatch (default: 32).
    replace : bool
        Sample with replacement from X (default: True). If False, note that each minibatch
        is sampled without replacement independently (so duplicates across minibatches can still occur).
    device : torch.device | None
        Device for running the networks. If None, inferred from each network's parameters.
    seed : int | None
        Seed for reproducible sampling.
    use_eval : bool
        If True, temporarily puts each network in eval() during scoring.

    Returns
    -------
    scores : np.ndarray
        Matrix of shape (len(A), len(B)), dtype float64.
    a_keys : np.ndarray
        The A keys in the row order used for 'scores'.
    b_keys : np.ndarray
        The B keys in the column order used for 'scores'.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (M, D). Got shape: {X.shape}")

    M, D = X.shape
    if K <= 0:
        raise ValueError("K must be > 0.")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be > 0.")

    # Stable ordering for matrix layout
    a_keys = np.array(sorted(A.keys()), dtype=int)
    b_keys = np.array(sorted(B.keys()), dtype=int)

    # RNG for reproducibility
    rng = np.random.default_rng(seed)

    # Pre-convert X to torch once (CPU), then index-select into it
    X_t = torch.from_numpy(X).float()  # shape (M, D), on CPU

    scores = np.empty((len(a_keys), len(b_keys)), dtype=np.float64)

    for i, ak in tqdm(enumerate(a_keys)):
        w = np.asarray(A[ak], dtype=np.float64).reshape(-1)
        if w.shape[0] != M:
            raise ValueError(f"A[{ak}] must have length M={M}. Got {w.shape[0]}.")
        if np.any(w < 0):
            raise ValueError(f"A[{ak}] contains negative weights.")
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError(f"A[{ak}] weights must sum to a positive finite value. Got {w_sum}.")

        p = w / w_sum  # normalize

        # Sample indices: shape (K, minibatch_size)
        # This samples each minibatch independently using the same distribution p.
        idx = rng.choice(
            M,
            size=(K, minibatch_size),
            replace=replace,
            p=p,
        )

        # Gather minibatches: shape (K, minibatch_size, D)
        idx_t = torch.from_numpy(idx).long()
        batch_t = X_t[idx_t]  # advanced indexing on CPU

        for j, bk in enumerate(b_keys):
            net = B[bk]

            # Infer device from network if not provided
            net_device = device
            if net_device is None:
                try:
                    net_device = next(net.parameters()).device
                except StopIteration:
                    net_device = torch.device("cpu")

            batch_in = batch_t.to(net_device, non_blocking=True)

            # Run forward pass
            prev_training = net.training
            if use_eval:
                net.eval()
            with torch.no_grad():
                out = net(batch_in)

                if not torch.is_tensor(out):
                    raise TypeError(
                        f"Network B[{bk}] returned type {type(out)}, expected a torch.Tensor."
                    )

                # Reduce to scalar: mean over all elements (covers (K, ...), any shape)
                score = out.float().mean().item()

            if use_eval and prev_training:
                net.train(True)

            scores[i, j] = score

    return scores, a_keys, b_keys



SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = False
# Create a simple model for demonstration
device = torch.device('cuda:0')
#load generator, load dataset, load weights
main_folder = "./saves_week29/"
files = os.listdir(main_folder)
saved_every = 30

embed_name = "embedding_dict_0_.pikl"
data_name = "data_0.npz"
one_hot_data_name = "one_hot_data_0.npz"
all_weights = {}
all_critics = {}

data = np.load(os.path.join(main_folder,data_name))
one_hot_data = np.load(os.path.join(main_folder, one_hot_data_name))
x = one_hot_data['x']
y = one_hot_data['y']



for i in range(0,500,saved_every):
    critic_checkpoint = torch.load(os.path.join(main_folder,"critic_checkpoint_"+str(i)+".pt"), map_location="cpu")
    config = critic_checkpoint["config"]
    weights_name = "weights_"+str(i)+".npz"
    all_rngs = []
    fixed_seed = np.random.randint(1e6)
    for _ in range(1):
        all_rngs.append(set_seed(fixed_seed,
                                device,
                            data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                            data_gen =SAME_DATA_SEEN,
                            network_init=SAME_NETWORK_INIT,))
    '''
    rngs,
    num_features,
    dropout,
    layers,
    embedding_dict,
    hidden_dim=1024,):
    '''
    critic = DeepSetCritic(rngs=all_rngs[0],
                                num_features = config["num_features"],
                                dropout = config["dropout"],
                                layers = config["layers"],
                                embedding_dict=config['embedding_dict']).to(device)
    missing, unexpected = critic.load_state_dict(critic_checkpoint["state_dict"], strict=False)

    gen_weights = np.load(os.path.join(main_folder,weights_name))['w'].flatten()
    
    all_critics[i] = critic
    all_weights[i] = gen_weights
    break


out = pairwise_weighted_minibatch_scores(
    A= all_weights,
    B= all_critics,
    X= x.squeeze(),
    K= 10,
    minibatch_size=32,
    device=torch.device('cuda:0'),
    seed = 0,
    use_eval = True,)

print(out[0])
#print(out[0])
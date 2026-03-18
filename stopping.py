import pickle
import numpy as np
import torch
import pandas as pd

from Discriminator import *
from util import set_seed
from typing import Dict, Hashable, Tuple, Optional, Any
from tqdm import tqdm

def pairwise_weighted_minibatch_scores(
    A: Dict[int, np.ndarray],
    B: Dict[int, nn.Module],
    X: np.ndarray,
    gt_dataset: np.ndarray,
    embedding_dict: Dict, 
    K: int,
    minibatch_size: int = 32,
    *,
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

    #iterate over all weight strategies
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
        batch_t = []
        for k in range(K):
            idx = rng.choice(
                M,
                size=(minibatch_size),
                replace=False,
                p=p,
            )

            # Gather minibatches: shape (K, minibatch_size, D)
            idx_t = torch.from_numpy(idx).long()
            batch_t.append(X_t[idx_t])  # advanced indexing on CPU
        batch_t = torch.stack(batch_t,dim=0)

        for j, bk in enumerate(b_keys):
            net = B[bk]
            if use_eval:
                net.set_eval()
            # Infer device from network if not provided
            net_device = device
            if net_device is None:
                try:
                    net_device = next(net.parameters()).device
                except StopIteration:
                    net_device = torch.device("cpu")

            batch_in = batch_t.to(net_device, non_blocking=True)

            # Run forward pass

            with torch.no_grad():
                out = net(batch_in)

                if not torch.is_tensor(out):
                    raise TypeError(
                        f"Network B[{bk}] returned type {type(out)}, expected a torch.Tensor."
                    )

                # Reduce to scalar: mean over all elements (covers (K, ...), any shape)
                score = out.float().mean().item()

            scores[i, j] = score

    for j, bk in enumerate(b_keys):
        net = B[bk]
        if use_eval:
            net.set_eval()
        index = np.random.choice(np.arange(len(gt_dataset)),size=minibatch_size*K,
                                       replace=True)
        sampled_data = torch.tensor(gt_dataset[index])
        sampled_data = embed_data(None,embedding_dict,sampled_data,overrite_start_idx=0)
        sampled_data = torch.reshape(sampled_data, (K, minibatch_size, X.shape[1])).to(device).float()
        with torch.no_grad():
            out = net(sampled_data)

            if not torch.is_tensor(out):
                raise TypeError(
                    f"Network B[{bk}] returned type {type(out)}, expected a torch.Tensor."
                )

            # Reduce to scalar: mean over all elements (covers (K, ...), any shape)
            score = out.float().mean().item()
            scores[:,j] -= score
    scores *= -1 #want to compute E[Truth score] - E[Fake Score] but rn its hte other way around
    return scores, a_keys, b_keys

def rescale_01(M):
    M = np.asarray(M, dtype=float)
    m_min = M.min()
    m_max = M.max()

    if m_max == m_min:
        return np.zeros_like(M)

    return (M - m_min) / (m_max - m_min)

def exp3_ix_selfplay_2p0s(
    L: np.ndarray,
    T: int = 100000,
    *,
    eta1: Optional[float] = None,
    gamma1: Optional[float] = None,
    eta2: Optional[float] = None,
    gamma2: Optional[float] = None,
    seed: Optional[int] = None,
    return_history: bool = False,
) -> Dict[str, Any]:
    """
    Simultaneous no-regret learning for both players in a 2-player zero-sum matrix game
    using EXP3-IX (bandit feedback).

    Inputs
    ------
    L : (n1, n2) ndarray
        Loss matrix for Player 1. Must have entries in [0, 1].
        Player 2's loss is defined as: L2 = 1 - L (per your spec).
    T : int
        Number of rounds.

    EXP3-IX (losses)
    ---------------
    - Maintain weights w over actions.
    - Play action a ~ p where p = w / sum(w).
    - Observe loss l(a) (bandit feedback).
    - Form implicit-exploration estimator:
          \hat{l}_a = l(a) / (p_a + gamma)
          \hat{l}_i = 0 for i != a
    - Update:
          w_a <- w_a * exp(-eta * \hat{l}_a)

    Returns
    -------
    dict with:
      - "p1_avg": average action distribution over time for player 1 (n1,)
      - "p2_avg": average action distribution over time for player 2 (n2,)
      - "p1_last": final round's distribution for player 1
      - "p2_last": final round's distribution for player 2
      - (optional) "history": per-round sampled actions & losses
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        raise ValueError(f"L must be 2D (n1, n2). Got shape {L.shape}")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("All entries of L must be in [0, 1].")

    n1, n2 = L.shape
    if T <= 0:
        raise ValueError("T must be > 0.")

    rng = np.random.default_rng(seed)

    # Reasonable defaults (tunable). Keep them small to avoid numerical issues.
    if eta1 is None:
        eta1 = np.sqrt(2.0 * np.log(max(n1, 2)) / (T * max(n1, 1)))
    if gamma1 is None:
        gamma1 = 0.5 * eta1

    if eta2 is None:
        eta2 = np.sqrt(2.0 * np.log(max(n2, 2)) / (T * max(n2, 1)))
    if gamma2 is None:
        gamma2 = 0.5 * eta2

    # Weights
    w1 = np.ones(n1, dtype=float)
    w2 = np.ones(n2, dtype=float)

    # Running average of distributions (time-average mixed strategy)
    p1_sum = np.zeros(n1, dtype=float)
    p2_sum = np.zeros(n2, dtype=float)

    # Optional history
    if return_history:
        a1_hist = np.empty(T, dtype=int)
        a2_hist = np.empty(T, dtype=int)
        l1_hist = np.empty(T, dtype=float)
        l2_hist = np.empty(T, dtype=float)

    def safe_probs(w: np.ndarray) -> np.ndarray:
        w = np.maximum(w, 1e-300)  # prevent zeros
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback: uniform
            return np.full_like(w, 1.0 / w.size, dtype=float)
        return w / s

    for t in tqdm(range(T)):
        # Current mixed strategies
        p1 = safe_probs(w1)
        p2 = safe_probs(w2)

        # Accumulate for average strategy
        p1_sum += p1
        p2_sum += p2

        # Sample actions
        a1 = rng.choice(n1, p=p1)
        a2 = rng.choice(n2, p=p2)

        # Realized losses (bandit feedback)
        l1 = L[a1, a2]
        l2 = 1.0 - l1  # per your spec for 2p0s

        # EXP3-IX implicit exploration estimators
        # (only the played action gets a nonzero estimate)
        est1 = l1 / (p1[a1] + gamma1)
        est2 = l2 / (p2[a2] + gamma2)

        # Weight updates (loss setting => negative exponent)
        w1[a1] *= np.exp(-eta1 * est1)
        w2[a2] *= np.exp(-eta2 * est2)

        if return_history:
            a1_hist[t] = a1
            a2_hist[t] = a2
            l1_hist[t] = l1
            l2_hist[t] = l2

    p1_last = safe_probs(w1)
    p2_last = safe_probs(w2)

    out: Dict[str, Any] = {
        "p1_avg": p1_sum / T,
        "p2_avg": p2_sum / T,
        "p1_last": p1_last,
        "p2_last": p2_last,
        "params": {
            "T": T,
            "eta1": float(eta1),
            "gamma1": float(gamma1),
            "eta2": float(eta2),
            "gamma2": float(gamma2),
        },
    }

    if return_history:
        out["history"] = {
            "a1": a1_hist,
            "a2": a2_hist,
            "l1": l1_hist,
            "l2": l2_hist,
        }

    return out


SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = False
# Create a simple model for demonstration
device = torch.device('cuda:0')
#load generator, load dataset, load weights
save_folder = "./saves_week29/"
folders = os.listdir(save_folder)
saved_every = 20

gt_data = (pd.read_csv('./data/censusHouseholdPulse_data/cleaned/ipums_cleaned_combined.csv').to_numpy(dtype=float, na_value=0))[:,1:]

all_results = {}
for counter, f in enumerate(folders):
    main_folder = os.path.join(save_folder,f)
    trial_id = int(f.split(":")[1])

    embed_name = "embedding_dict_"+str(trial_id)+"_.pikl"
    with open(os.path.join(main_folder,embed_name), 'rb') as file:
        embed_dict = pickle.load(file)

    data_name = "data_"+str(trial_id)+".npz"
    one_hot_data_name = "one_hot_data_"+str(trial_id)+".npz"
    all_weights = {}
    all_critics = {}

    data = np.load(os.path.join(main_folder,data_name))
    one_hot_data = np.load(os.path.join(main_folder, one_hot_data_name))
    x = one_hot_data['x']
    y = one_hot_data['y']

    for i in range(0,600,saved_every):
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


    out = pairwise_weighted_minibatch_scores(
        A= all_weights,
        B= all_critics,
        X= x.squeeze(),
        gt_dataset=gt_data,
        embedding_dict=embed_dict,
        K= 20,
        minibatch_size=128,
        device=torch.device('cuda:0'),
        seed = 0,
        use_eval = True,)
    
    weight_start_idx = 2
    nn_start_idx = 1
    conf_matrix = out[0][weight_start_idx:,nn_start_idx:]
    conf_matrix = rescale_01(conf_matrix)

    out = exp3_ix_selfplay_2p0s(L=conf_matrix)
    # print(out['p1_avg'])

    sum_pred = 0
    for i in range(out['p1_avg'].shape[0]):
        weights_idx = (weight_start_idx + i)*20
        sum_pred += (all_weights[weights_idx] @ y)*out['p1_avg'][i]

    highest_idx = np.argmax(out['p1_avg'])
    highest_true = (highest_idx + weight_start_idx)*20

    highest_nn_idx = np.argmax(out['p2_avg'])
    highest_nn_true = (highest_nn_idx + nn_start_idx)*20

    all_results[f] = {}
    all_results[f]['last'] = all_weights[580] @ y
    all_results[f]['avg'] = sum_pred
    all_results[f]['best'] = all_weights[highest_true] @ y
    all_results[f]['idx_info'] = [highest_idx, highest_true]
    all_results[f]['nn_idx'] = [highest_nn_idx, highest_nn_true]

for key in all_results:
    print(key)
    print(all_results[key])
    print("")
#print(out[0])
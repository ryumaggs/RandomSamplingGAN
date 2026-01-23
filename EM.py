import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
def em_reweight_x1_to_x2_unique(
        X1, X2u, p2, Cs,
        alpha=0.95,
        max_iter=100,
        tol=1e-7,
        chunk_size=256,
        prior_eps=1e-12,
    ):
    """
    EM to find weights on X1 (atoms) so that mixture with categorical kernel matches X2 unique distribution.

    Returns:
        pi: (N1,) mixture weights summing to 1
        w:  (N1,) survey-style weights with mean ~ 1 (sum = N1)
    """
    X1 = np.asarray(X1, dtype=np.int64)
    X2u = np.asarray(X2u, dtype=np.int64)
    p2 = np.asarray(p2, dtype=np.float64)

    N1, D = X1.shape
    M = X2u.shape[0]
    assert X2u.shape[1] == D
    assert p2.shape == (M,)

    # Normalize p2 to sum to 1 (safe even if already probs; if counts, becomes probs)
    p2 = p2 / p2.sum()

    # alpha per-dim
    if np.isscalar(alpha):
        alpha = np.full(D, float(alpha))
    else:
        alpha = np.asarray(alpha, dtype=np.float64)
        assert alpha.shape == (D,)

    Cs = np.asarray(Cs, dtype=np.int64)
    assert Cs.shape == (D,)

    # Precompute per-dim log probs for match / mismatch
    # match: alpha + (1-alpha)/C
    # mismatch: (1-alpha)/C
    log_p_match = np.log(alpha + (1.0 - alpha) / Cs)
    log_p_mism  = np.log((1.0 - alpha) / Cs)

    # init pi uniform
    pi = np.full(N1, 1.0 / N1, dtype=np.float64)
    log_pi = np.log(pi)

    for it in tqdm(range(max_iter)):
        pi_old = pi.copy()

        # Accumulator for M-step: s_i = sum_u p2[u] * r_ui
        s = np.zeros(N1, dtype=np.float64)

        # We compute r_ui in chunks over i to avoid building full MxN1 matrix
        # For each u, denominator needs sum_i pi_i K(u,i).
        # We'll compute denominators via log-sum-exp over i in streaming chunks.

        # 1) Compute log_den[u] = log sum_i exp(log_pi[i] + logK[u,i])
        log_den = np.full(M, -np.inf, dtype=np.float64)

        for start in range(0, N1, chunk_size):
            end = min(N1, start + chunk_size)
            Xi = X1[start:end]  # (B, D)
            # match matrix: (M, B, D) would be huge, so do per-dim accumulation
            # logK[u,b] = sum_d log_p_match[d] if equal else log_p_mism[d]
            logK = np.zeros((M, end - start), dtype=np.float64)
            for d in range(D):
                eq = (X2u[:, [d]] == Xi[None, :, d])  # (M, B)
                logK += np.where(eq, log_p_match[d], log_p_mism[d])

            # add log_pi
            chunk_log = logK + log_pi[start:end][None, :]

            # stable log-sum-exp combine with running log_den
            m = np.maximum(log_den, np.max(chunk_log, axis=1))
            log_den = m + np.log(np.exp(log_den - m) + np.sum(np.exp(chunk_log - m[:, None]), axis=1))

        # 2) Now accumulate s_i = sum_u p2[u] * r_ui
        # r_ui = exp(log_pi[i] + logK[u,i] - log_den[u])
        for start in range(0, N1, chunk_size):
            end = min(N1, start + chunk_size)
            Xi = X1[start:end]
            logK = np.zeros((M, end - start), dtype=np.float64)
            for d in range(D):
                eq = (X2u[:, [d]] == Xi[None, :, d])
                logK += np.where(eq, log_p_match[d], log_p_mism[d])

            log_r = (log_pi[start:end][None, :] + logK) - log_den[:, None]
            r = np.exp(log_r)  # (M, B)
            # weighted sum over u
            s[start:end] = (p2[:, None] * r).sum(axis=0)

        # M-step with tiny prior to prevent exact zeros
        pi = s + prior_eps
        pi /= pi.sum()
        log_pi = np.log(pi)

        # convergence
        delta = np.max(np.abs(pi - pi_old))
        if delta < tol:
            break

    # Convert to survey-like weights with mean 1 (sum = N1)
    w = pi * N1
    return pi, w
    
if __name__ == "__main__":
    
    if False:
        with open("./EMPredictions.pikl", 'rb') as file:
            tobj = pickle.load(file)
        print(tobj)
        exit(1)
    
    ground_truth_path='./data/HouseholdPulse_data/cleaned/ipums_cleaned_combined.csv'
    
    
    gt_df = pd.read_csv(ground_truth_path)
    
    X2u = gt_df.to_numpy()[:,1:]
    p2 = gt_df.to_numpy()[:,0]
    Cs = gt_df.iloc[:,1:].nunique().tolist()
    all_predictions={}
    for week_int in range(22,30):
        week = str(week_int)
        bias_path = './data/HouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv'
        bi_df = pd.read_csv(bias_path)
        bi_df_sampled = bi_df.sample(n=2500)
        X1 = bi_df_sampled.to_numpy()[:,1:]
        Y = bi_df_sampled.to_numpy()[:,0]
        
        alpha=0.95
        pi, w = em_reweight_x1_to_x2_unique(
            X1, X2u, p2, Cs,
            alpha=alpha,
            max_iter=100,
            tol=1e-7,
            chunk_size=256,
            prior_eps=1e-12,)
        print(np.sum(Y * pi))
        
        print("predicted vax~: ", np.sum(Y * pi))
        all_predictions[week_int]= np.sum(Y*pi)
        
    with open("./EMPredictions.pikl", 'wb') as file:
        pickle.dump(all_predictions,file)
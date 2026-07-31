#Global imports
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
from scipy.stats import spearmanr
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import pickle
import torch
import re
import ast

def plot_2d_runs(data):
    # Compute mean and standard deviation across experiments
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)  # Or use std / sqrt(n) for standard error

    # Time steps (x-axis)
    time_steps = np.arange(data.shape[1])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, mean, label='Mean Time Series')
    plt.fill_between(time_steps, mean - std, mean + std, alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Average Time Series with Error Bands')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_runs_as_numpy(runs_path, var_names, filter_by, num_runs=9):
    run_folders = []
    count = 0
    for name in os.listdir(runs_path):
        valid_file = True
        for fb in filter_by:
            if fb not in name:
                valid_file = False
                break
        if not valid_file:
            continue
        if os.path.isdir(os.path.join(runs_path,name)):
            run_folders.append(os.path.join(runs_path,name))
            count += 1
        if count >= num_runs:
            break

    event_files = []
    for run_folder in run_folders:
        for name in os.listdir(run_folder):
            if os.path.isdir(os.path.join(run_folder,name)):
                continue
            else:
                event_files.append(os.path.join(run_folder,name))
    all_data = {}
    for i_name, vname in tqdm(enumerate(var_names)):
        all_data[vname] = []
        for i, event_file in enumerate(event_files):
            # Load event accumulator
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # List available tags (scalars, histograms, images, etc.)
            # Read scalar values (e.g., 'loss', 'accuracy')
            if not vname in ea.Tags()["scalars"]:
                continue
            prediction_events = ea.Scalars(vname)
            #summ = 0
            #for iter, event in enumerate(prediction_events):
            #    print(event, l2_events[iter])
            cur_pred = []
            for event in prediction_events:
                cur_pred.append(event.value)
            all_data[vname].append(cur_pred)
        lengths = set(len(x) for x in all_data[vname])
        if len(lengths) > 1:
            print(f"MISMATCH in {vname}: {lengths}")
        all_data[vname] = np.array(all_data[vname])
    return all_data, event_files

def load_all_as_dict(runs_path, var_names, filter_by):
    run_folders = []
    count = 0
    all_names = []
    for name in os.listdir(runs_path):
        if os.path.isdir(os.path.join(runs_path,name)):
            run_folders.append(os.path.join(runs_path,name))
            all_names.append(name)
            count += 1

    event_files = []
    for run_folder in run_folders:
        for name in os.listdir(run_folder):
            if os.path.isdir(os.path.join(run_folder,name)):
                continue
            else:
                event_files.append(os.path.join(run_folder,name))

    all_data = {}
    for i_name, vname in enumerate(var_names):
        all_data[vname] = []
        for i, event_file in enumerate(event_files):
            # Load event accumulator
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # List available tags (scalars, histograms, images, etc.)
            # Read scalar values (e.g., 'loss', 'accuracy')
            prediction_events = ea.Scalars(vname)
            #summ = 0
            #for iter, event in enumerate(prediction_events):
            #    print(event, l2_events[iter])
            cur_pred = []
            for event in prediction_events:
                cur_pred.append(event.value)
            all_data[vname].append(cur_pred)
        all_data[vname] = np.array(all_data[vname])
    return all_data, all_names

def print_stats(runs_path):
    all_data = load_runs_as_numpy(runs_path, ['Vaccine prediction', 'l2 norm demo diff'])
    all_vac_predictions = all_data['Vaccine prediction']
    all_l2_demographics = all_data['l2 norm demo diff']
    vac_stds = np.std(all_vac_predictions,axis=0)
    l2_stds = np.std(all_l2_demographics,axis=0)
    print("Consistency of Vac | consistency of L2 (30 trials)")
    print(np.mean(vac_stds),np.mean(l2_stds))
    print("")

    print("Lowest point of vacc variance")
    print(np.argmin(vac_stds[25:]), np.min(vac_stds[25:]), np.mean(all_vac_predictions[:,np.argmin(vac_stds[25:])]))

    #plot_2d_runs(all_vac_predictions[:,25:])

def find_peaks_troughs(runs_path, filter_by):
    sigma = 25  # smoothing level
    mr = 30
    out = load_runs_as_numpy(runs_path,  
                             ['Vaccine prediction total', 'GenLoss', 'Gen Entropy'],
                             filter_by=filter_by)
    vpt = out['Vaccine prediction total']
    gloss = out['GenLoss']
    entropy = out['Gen Entropy']
    for i, y in enumerate(gloss):
        # Smooth the signal
        smoothed = gaussian_filter1d(y, sigma=sigma)
            # Find peaks and troughs
        peaks, _ = find_peaks(smoothed)
        troughs, _ = find_peaks(-smoothed)

        converted_peaks = [int(p/2) for p in peaks]
        converted_troughs = [int(t/2) for t in troughs]
        #print(vpt.shape, converted_peaks, converted_troughs)
        peak_means = [np.mean(vpt[i,max(cp-mr,0):min(cp+mr,vpt.shape[1])]) for cp in converted_peaks]
        trough_means = [np.mean(vpt[i,max(ct-mr,0):min(ct+mr,vpt.shape[1])]) for ct in converted_troughs]

        print(np.mean(vpt[i, :]), trough_means)

def test(runs_path, filter_by):
    sigma = 25  # smoothing level
    mr = 30
    out = load_runs_as_numpy(runs_path,  
                             ['Vaccine prediction total', 'GenLoss', 'Gen Entropy'],
                             filter_by=filter_by)
    vpt = out['Vaccine prediction total']
    gloss = out['GenLoss']
    entropy = out['Gen Entropy']

    gloss_norms = np.linalg.norm(gloss, axis=1, keepdims=True)
    normalized_gloss = gloss / gloss_norms

    entropy_norms = np.linalg.norm(entropy, axis=1, keepdims=True)
    normalized_entropy = entropy / entropy_norms
    summed_arr = np.zeros(vpt.shape)
    for i, y in enumerate(normalized_entropy):
        for j in range(len(y)):
            summed_arr[i,j] = y[j] #normalized_gloss[i, j*2] + y[j]

        smoothed = gaussian_filter1d(summed_arr[i,:], sigma=sigma)
        mins = [np.argmin(smoothed)]

        trough_means = [np.mean(vpt[i,max(ct-mr,0):min(ct+mr,vpt.shape[1])]) for ct in mins]
        print(trough_means)
        print("")

    #idea 1 add gloss to entropy and find the point at which they are the lowest

def find_trend_plateau(y, smooth_sigma=20, threshold=1e-4):
    """
    Detects where a decreasing curve stops declining and starts plateauing or rising.
    
    Parameters:
        y (np.ndarray): 1D signal (e.g., loss curve).
        smooth_sigma (float): Smoothing factor for Gaussian smoothing.
        threshold (float): Derivative threshold to detect plateau or increase.

    Returns:
        (int, float): Index and original value at the turning/plateau point.
    """
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)

    dy = np.gradient(y_smooth)

    for i in range(1, len(dy)):
        if dy[i] > threshold and dy[i-1] <= threshold:
            return i, y[i], y_smooth  # return index in original array

    # Fallback: return global min if no trend reversal found
    min_idx = np.argmin(y_smooth)

    return min_idx, y[min_idx], y_smooth

def compute_rolling_std_numpy(y, window=20):
    if window < 1 or window > len(y):
        raise ValueError("Invalid window size")

    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode='reflect')  # or 'edge'

    rolling_std = np.empty(len(y))
    for i in range(len(y)):
        window_slice = padded[i:i + window]
        rolling_std[i] = np.std(window_slice)

    return rolling_std

def normalize_0_1(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def x_y_relationship(out):
    start_point = 0
    end_point = 300
    Y = out['EMPSTAT prediction'][:,start_point:end_point]
    gloss = out['GenLoss'][:,start_point:end_point]
    entropy = out['Gen Entropy'][:,start_point:end_point]
    bscore = out['Bias score'][:,start_point:end_point]
    gp = out['Gradient Penalty'][:,start_point:end_point]

    X = -1 * np.stack([gloss, entropy, bscore, gp])
    V = 0.6

    N, T, M = Y.shape[0], Y.shape[1], X.shape[0]
    if 'metric_names' not in globals():
        metric_names = [f"X_{j+1}" for j in range(M)]

    # ---------------------------------------------------------
    # 1. Spearman correlation with closeness measure exp(-|Y - V|)
    # ---------------------------------------------------------
    rho = np.zeros((M, T))
    for j in range(M):
        for t in range(T):
            closeness = np.exp(-np.abs(Y[:, t] - V))
            rho[j, t], _ = spearmanr(X[j, :, t], closeness)

    # ---------------------------------------------------------
    # 2. Spearman correlation with absolute error (negative)
    # ---------------------------------------------------------
    rho_err = np.zeros((M, T))
    for j in range(M):
        for t in range(T):
            abs_err = -np.abs(Y[:, t] - V)
            rho_err[j, t], _ = spearmanr(X[j, :, t], abs_err)

    # ---------------------------------------------------------
    # 3. Hybrid correlation × accuracy measure
    # ---------------------------------------------------------
    mean_abs_err = np.mean(np.abs(Y - V), axis=0)  # average error per time step
    norm_err = mean_abs_err / mean_abs_err.max()   # normalize 0–1
    H = rho * (1 - norm_err)                       # correlation weighted by accuracy

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    titles = [
        r"(a) Spearman $\rho(X, e^{-|Y-V|})$",
        r"(b) Spearman $\rho(X, -|Y-V|)$",
        r"(c) Hybrid $ \rho \times (1 - \mathrm{norm\ error})$"
    ]
    data_list = [rho, rho_err, H]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Time step (t)")
        ax.set_ylabel("Aux variable (j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
        
def oscillation_range_normalized(matrix):
    """MAD normalized by per-row range, then averaged."""
    ranges = np.ptp(matrix, axis=1)  # per-row (max-min)
    # avoid div-by-zero by flooring range
    ranges = np.maximum(ranges, 1e-12)
    per_row_mad = np.mean(np.abs(np.diff(matrix, axis=1)), axis=1)
    return np.mean(per_row_mad / ranges), np.mean(matrix,axis=1)

def success_rate_by_window_comparison(runs_path, filter_by, num_runs, window_length):
    """
    Compares the average of the first K and last K points for each row in a 2D array.

    Parameters:
    - loss_array: np.ndarray of shape (num_runs, time_steps)
    - K: int, number of points to average at the start and end

    Returns:
    - success_percent: float, percentage of rows where avg(first K) > avg(last K)
    """
    out = load_runs_as_numpy(runs_path,  
                             ['tvd', 'GenLoss', 'Gen Entropy'],
                             filter_by=filter_by,
                             num_runs=num_runs)
    loss_array = out['tvd']
    if window_length <= 0 or window_length > loss_array.shape[1] // 2:
        raise ValueError("K must be > 0 and <= half the number of time steps per row")

    start_avg = np.mean(loss_array[:, :window_length], axis=1)
    end_avg = np.mean(loss_array[:, -window_length:], axis=1)
    success_mask = start_avg > end_avg
    success_percent = 100 * np.mean(success_mask)

    return success_percent

def success_rate_JSD_loss(runs_path, filter_by, num_runs, window_length,override_loss_array=None):
    if override_loss_array is None:
        out = load_runs_as_numpy(runs_path,  
                                ['tvd', 'GenLoss', 'Gen Entropy'],
                                filter_by=filter_by,
                                num_runs=num_runs)
        loss_array = out['tvd']
    else:
        loss_array = override_loss_array
    nearly_identical = return_success_proportions(loss_array, 
                        0.03,
                        window_length)
    excellent = return_success_proportions(loss_array, 
                        0.05,
                        window_length)
    very_similar = return_success_proportions(loss_array, 
                        0.07,
                        window_length)
    print("Nearly Identical (0.03): ", nearly_identical)
    print("")
    print("Excellent (0.05): ", excellent)
    print("")
    print("Very similar (0.07): ", very_similar)
    return loss_array

def return_success_proportions(loss_array, 
                               threshold,
                               window_length):
    end_avg = np.mean(loss_array[:, -window_length:], axis=1)
    success_mask = end_avg <= threshold
    success_percent = 100 * np.mean(success_mask)
    mask = np.any(loss_array < threshold, axis=1)
    # Calculate percentage
    percent = 100 * np.sum(mask) / loss_array.shape[0]
    return success_percent, percent

def create_joint_table(runs_path, 
                       threshold,
                       filter_by,
                       var_list):
    out, names = load_all_as_dict(runs_path,  
                                ['tvd', 'GenLoss', 'Gen Entropy'],
                                filter_by=filter_by,)
    loss_array = out['tvd']

    results_dict = {}
    for name, loss in zip(names, loss_array):
        #determine what vars are in the name
        print(loss[0])
        var_count = 0
        c_vars = []
        for v in var_list:
            if v in name:
                var_count += 1
                c_vars.append(v)
        if len(c_vars) > 2:
            print(c_vars)
            print("ERROR: should only be 2 variables")
            assert 1 == 0
        c_vars = sorted(c_vars)

        #create results key
        new_key = tuple(c_vars)
        new_result = int(np.any(loss < threshold, axis=0))

        #store result in list
        if new_key not in results_dict:
            results_dict[new_key] = []
        results_dict[new_key].append(new_result)
    data = results_dict
    unique_strings = sorted(set([s for pair in data.keys() for s in pair]))

    # Create an empty DataFrame
    table = pd.DataFrame(index=unique_strings, columns=unique_strings, dtype=float)

    # Fill in the DataFrame with averages
    for (x, y), values in data.items():
        avg = np.mean(values)
        table.loc[x, y] = avg

    print(table)
    return results_dict, table

def HHP_load_all_weeks_as_dict(runs_path):
    to_load = [str(i) for i in range(23,30)]
    to_load = to_load + ["ALL"]
    all_outs = {}
    for key in to_load:
        out, _ = load_runs_as_numpy(runs_path,  
                                ['target prediction', 'GenLoss', 'Gen Entropy', 
                                'tvd', 'Warmup', 'Bias score', 'Gradient Penalty'],
                                filter_by=["Week="+str(key)],
                                num_runs=20)
        all_outs[key] = out
    return all_outs

def process_load_output(out, smoothing_window):
    start_point = 2
    end_point = 500
    Y = out['EMPSTAT prediction'][:,start_point:end_point]
    gloss = out['GenLoss'][:,start_point:end_point]
    entropy = out['Gen Entropy'][:,start_point:end_point]
    bscore = out['Bias score'][:,start_point:end_point]
    gp = out['Gradient Penalty'][:,start_point:end_point]

    X = np.stack([gloss, entropy, bscore, gp])
    N, T, M = Y.shape[0], Y.shape[1], X.shape[0]
    #V = 0.6
    #t_star = np.argmin(np.abs(Y - V), axis=1)   # shape (N,)
    ### analysis of variables down below
    smooth_Y = True
    # Smooth Y if desired
    if smooth_Y:
        Y_smooth = uniform_filter1d(Y, size=smoothing_window, axis=1)
    else:
        Y_smooth = Y
    
    return N, T, M, Y, Y_smooth, X

def process_all_weeks_dict(all_outs,
                           save_dir,
                           save=False,):
    for key in all_outs:
        if key == 'ALL':
            continue
        entropy = all_outs[key]['Gen Entropy']
        vpt = all_outs[key]['target prediction']
        print(entropy.shape)
        all_min_entropy_idx = np.argmin(entropy,axis=1)
        all_min_entropy_preds = []
        for i, amei in enumerate(all_min_entropy_idx):
            all_min_entropy_preds.append(vpt[i,amei])
        all_min_entropy_preds = np.array(all_min_entropy_preds)
        sorted_min_entropy_preds = np.sort(all_min_entropy_preds)

        #isolate out the median and which trial it came from
        median = sorted_min_entropy_preds[(sorted_min_entropy_preds.shape[0]-1)//2]
        selected_trial = np.where(all_min_entropy_preds == median)[0].item()
        print(key, median)
        if save:
            #load in the weights and pick out the corresponding weight
            dirr = os.path.join(save_dir,str(key))
            data = np.load(dirr+"/data_"+str(selected_trial)+".npz")
            weights = np.load(dirr+"/weight_history_"+str(selected_trial)+".npz")
            datum = data['x']
            labels = data['y']
            weights = weights['w']
            pws = ((weights @ labels).T.flatten())
            #print(datum.shape, labels.shape)
            closest_time_step = np.where(np.isclose(pws,median))[0].item()
            selected_weights = weights[closest_time_step,:]
            if save:
                np.savez(save_dir+"/chosen_weight.npz", w=selected_weights)
                np.savez(save_dir+"/chosen_data.npz", x = datum, y = labels)
                np.savez(save_dir+"/chosen_weight_history.npz", w = weights)

def absolute_analysis(out):
    top_K = 5
    smoothing_window = 5
    N, T, M, Y, Y_smooth, X = process_load_output(out, smoothing_window)
    Y_max_topK_agg = np.zeros(M)
    Y_min_topK_agg = np.zeros(M)

    for j in range(M):
        all_x_max = []
        all_Y_max = []
        all_x_min = []
        all_Y_min = []

        for n in range(N):
            # Smooth auxiliary variable
            x_smooth = uniform_filter1d(X[j, n, :], size=smoothing_window)

            # ----------------- maxima -----------------
            max_val = np.max(x_smooth)
            max_idx = np.argmax(x_smooth)
            all_x_max.append(max_val)
            all_Y_max.append(Y_smooth[n, max_idx])

            # ----------------- minima -----------------
            min_val = np.min(x_smooth)
            min_idx = np.argmin(x_smooth)
            all_x_min.append(min_val)
            all_Y_min.append(Y_smooth[n, min_idx])

        # ----------------- aggregate top K across all experiments -----------------
        # maxima
        all_x_max = np.array(all_x_max)
        all_Y_max = np.array(all_Y_max)
        top_indices = np.argsort(all_x_max)[-top_K:]  # top K largest values
        Y_max_topK_agg[j] = np.mean(all_Y_max[top_indices])

        # minima
        all_x_min = np.array(all_x_min)
        all_Y_min = np.array(all_Y_min)
        top_indices = np.argsort(all_x_min)[:top_K]  # top K smallest values
        Y_min_topK_agg[j] = np.mean(all_Y_min[top_indices])

    # Example access
    for j in range(M):
        print(f"Aux {j}: Mean Y at top {top_K} maxima across all experiments = {Y_max_topK_agg[j]:.3f}")
        print(f"Aux {j}: Mean Y at top {top_K} minima across all experiments = {Y_min_topK_agg[j]:.3f}")
        print("")

def peak_analysis(out):
    '''
    out - output from load_numpy runs


    X - np array auxilery m etrics
    Y - prediction variable
    V - true value
    '''
    top_K = 5  # number of extrema to select across all experiments
    smoothing_window = 3  # window size for uniform_filter1d
    start_point = 10
    end_point = 500
    Y = out['EMPSTAT prediction'][:,start_point:end_point]
    gloss = out['GenLoss'][:,start_point:end_point]
    entropy = out['Gen Entropy'][:,start_point:end_point]
    bscore = out['Bias score'][:,start_point:end_point]
    gp = out['Gradient Penalty'][:,start_point:end_point]

    X = np.stack([gloss, entropy, bscore, gp])
    N, T, M = Y.shape[0], Y.shape[1], X.shape[0]
    #V = 0.6
    #t_star = np.argmin(np.abs(Y - V), axis=1)   # shape (N,)
    ### analysis of variables down below

    # Optionally smooth Y
    Y_smooth = uniform_filter1d(Y, size=smoothing_window, axis=1)

    # ---------------------------------------------------------
    # Storage for results
    # ---------------------------------------------------------
    # Each entry is a list of Y values at extrema for that variable and experiment
    Y_at_max = [[[] for _ in range(N)] for _ in range(M)]
    Y_at_min = [[[] for _ in range(N)] for _ in range(M)]

    # ---------------------------------------------------------
    # Find local extrema
    # ---------------------------------------------------------
    for j in range(M):  # auxiliary variables
        for n in range(N):  # experiments
            # Smooth auxiliary variable
            x_smooth = uniform_filter1d(X[j, n, :], size=smoothing_window)
            
            # Find maxima
            peaks, _ = find_peaks(x_smooth)
            Y_at_max[j][n] = Y_smooth[n, peaks].tolist()
            
            # Find minima by inverting
            troughs, _ = find_peaks(-x_smooth)
            Y_at_min[j][n] = Y_smooth[n, troughs].tolist()

    # ---------------------------------------------------------
    # Example access
    # Y_at_max[0][0] -> list of Y values at local maxima of auxiliary variable 0 in experiment 0
    # Y_at_min[2][5] -> list of Y values at local minima of auxiliary variable 2 in experiment 5
    # ---------------------------------------------------------

    # Optional: print summary
    #for j in range(M):
    #    for n in range(N):
    #        print(f"Aux {j}, Exp {n}: {len(Y_at_max[j][n])} maxima, {len(Y_at_min[j][n])} minima")

    # Storage for aggregated Y at top K extrema across all experiments
    Y_max_topK_agg = np.zeros(M)
    Y_min_topK_agg = np.zeros(M)

    for j in range(M):
        # ----------------- maxima -----------------
        all_x_max = []
        all_Y_max = []
        for n in range(N):
            # Original peaks
            peaks, _ = find_peaks(uniform_filter1d(X[j, n, :], size=smoothing_window))
            if len(peaks) > 0:
                all_x_max.extend(X[j, n, peaks])
                all_Y_max.extend(Y_smooth[n, peaks])
        if len(all_x_max) > 0:
            all_x_max = np.array(all_x_max)
            all_Y_max = np.array(all_Y_max)
            top_indices = np.argsort(all_x_max)[-top_K:]  # top K largest peaks
            Y_max_topK_agg[j] = np.mean(all_Y_max[top_indices])
        else:
            Y_max_topK_agg[j] = np.nan

        # ----------------- minima -----------------
        all_x_min = []
        all_Y_min = []
        for n in range(N):
            troughs, _ = find_peaks(-uniform_filter1d(X[j, n, :], size=smoothing_window))
            if len(troughs) > 0:
                all_x_min.extend(X[j, n, troughs])
                all_Y_min.extend(Y_smooth[n, troughs])
        if len(all_x_min) > 0:
            all_x_min = np.array(all_x_min)
            all_Y_min = np.array(all_Y_min)
            top_indices = np.argsort(all_x_min)[:top_K]  # top K deepest troughs
            Y_min_topK_agg[j] = np.mean(all_Y_min[top_indices])
        else:
            Y_min_topK_agg[j] = np.nan

    # Example access
    for j in range(M):
        print(f"Aux {j}: Mean Y at top {top_K} maxima across all experiments = {Y_max_topK_agg[j]:.3f}")
        print(f"Aux {j}: Mean Y at top {top_K} minima across all experiments = {Y_min_topK_agg[j]:.3f}")
        print("")

def entropy_pred_relationship(runs_path, filter_by, num_runs,
                              target_var_name,
                              metric_tb_name,
                              K=5):
    out, event_files = load_runs_as_numpy(runs_path,  
                             [target_var_name, 'GenLoss', 'Gen Entropy', 
                              'tvd', 'Warmup', 'Bias score', 'Gradient Penalty',metric_tb_name],
                             filter_by=filter_by,
                             num_runs=num_runs,
                             )
    print("")
    print("-----------------------------------")
    print("Filter by: ", filter_by)
    vpt = out[target_var_name]
    gloss = out['GenLoss']
    entropy = out['Gen Entropy']
    warmup = out['Warmup']
    bscore = out['Bias score']
    gp = out['Gradient Penalty']
    tvd = out['tvd']
    metric = out[metric_tb_name]
    print(vpt.shape)
    #oscore, omean = oscillation_range_normalized(gp)
    #tvd_score, tvd_mean = oscillation_range_normalized(tvd)
    #print("Gradient Penalty: ")
    #print("oscillation  score: ", oscore)
    #print("average gp over time and exp: ", np.mean(omean))
    #print("TVD: ")
    #print("oscillation score: ", tvd_score)
    #print("average gp over time and exp: ", np.mean(tvd_mean))
    #indices_in_range = np.where((entropy[0] >= 0.75) & (entropy[0] <= 0.8))[0]
    #t_f_scores = out['Truth - Fake scores']
    
    order = np.arange(vpt.shape[0])
    start_point = 0
    end_point = 600 #len(entropy[0])
    #indices = np.arange(len(entropy[0,start_point:end_point]))
    labels = ['entropy', 'truth-fake', 'tvd']
    selected_x_axis = metric #entropy
    #vpt_b= vpt[:,start_point:end_point]
    #print((np.argmin(np.abs(vpt_b.mean(axis=0) - 0.6))) + start_point)
    #print((np.min(np.abs(vpt_b.mean(axis=0) - 0.6))))
    #if False: #old analysis 
    if True: #entropy
        for k in [K]: #[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            #k = 20  # number of smallest elements per row
            # Get indices of the k smallest elements per row
            idx = np.argpartition(selected_x_axis, k, axis=1)[:, :k]
            # Use advanced indexing to gather corresponding B values
            rows = np.arange(selected_x_axis.shape[0])[:, None]
            selected_B = vpt[rows, idx]
            #plt.scatter(selected_B, np.zeros_like(selected_B), alpha=0.6, s=20)
            #plt.show()
            # Compute row-wise means
            mean_B = np.mean(selected_B, axis=1)
            print(k, np.mean(mean_B),"|| +/- ",np.std(mean_B))
        print("")

        if False:
            fig, axs = plt.subplots(3, 3, figsize=(10, 10))
            for i in order[0:9]:
                name = ""
                ''' puts the week/file as the name
                parts = event_files[i].split("||", 2)
                if len(parts) >= 3:
                    name = parts[1]
                if i > vpt.shape[0]:
                    break
                '''
                #warmup_mean = np.mean(warmup[i,-50:])
                name = "" #str(round(warmup_mean,3))
                
                row, col = divmod(i, 3)
                minidx, _, smoothed = find_trend_plateau(vpt[i], smooth_sigma = 20)
                print(minidx, np.mean(vpt[i,max(0,minidx-2):minidx+2]))
                norm_entrop = normalize_0_1(entropy[i,start_point:end_point])
                norm_tvd = normalize_0_1(tvd[i,start_point:end_point])
                norm_score = normalize_0_1(bscore[i,start_point:end_point])
                norm_gp = normalize_0_1(gp[i,start_point:end_point])

                x_axis =  entropy[i,start_point:end_point]
                y_axis = vpt[i,start_point:end_point]
                sc = axs[row,col].scatter(x_axis,y_axis,c=indices, cmap='viridis')
                axs[row,col].set_title(name)
                #sc = axs[row+1,col].scatter(t_f_scores[i,start_point:end_point],vpt[i,start_point:end_point],c=indices, cmap='viridis')
                #sc = axs[row+2,col].scatter(tvd[i,start_point:end_point],vpt[i,start_point:end_point],c=indices, cmap='viridis')
                #fig.text(0.05, 0.85 - i * 0.3, labels[i], va='center', ha='right', fontsize=12)

            if sc is not None:
                cbar = fig.colorbar(sc, ax=axs, orientation='vertical', shrink=0.8)
                cbar.set_label('Time Index')

            return indices_in_range

    if False: #rate of change
        diffs = np.abs(np.diff(vpt))  # shape (N, T-1)
        K = 1e-6
        # Boolean mask where condition holds
        mask = diffs < K  # shape (N, T-1)

        # Find first True per row
        first_indices = np.argmax(mask, axis=1)

        # If a row never satisfies condition, argmax returns 0 incorrectly.
        # So we check which rows actually contain any True.
        has_valid = mask.any(axis=1)
        first_indices = np.where(has_valid, first_indices, -1)
        rows = np.arange(vpt.shape[0])
        print(first_indices)
        print("roc: ", np.mean(vpt[rows,first_indices]),np.std(vpt[rows,first_indices]))
    #data = np.load("./saves/data.npz")
    #weights = np.load("./saves/weight_history.npz")

    #labels = data['y']
    #weights = weights['w']
    #pws = (weights @ labels).T
    #print("diff: ", np.linalg.norm(pws-vpt))

def count_row_differences(X, Y):
    """
    Returns:
        x_not_in_y : number of rows in X that do not appear in Y
        y_not_in_x : number of rows in Y that do not appear in X
    """

    # Convert rows to structured dtype so NumPy can compare row-wise
    X_view = np.ascontiguousarray(X).view(
        np.dtype((np.void, X.dtype.itemsize * X.shape[1]))
    )
    Y_view = np.ascontiguousarray(Y).view(
        np.dtype((np.void, Y.dtype.itemsize * Y.shape[1]))
    )

    # Unique row sets
    X_unique = np.unique(X_view)
    Y_unique = np.unique(Y_view)

    # Set differences
    x_not_in_y = np.setdiff1d(X_unique, Y_unique).shape[0]
    y_not_in_x = np.setdiff1d(Y_unique, X_unique).shape[0]

    return x_not_in_y, y_not_in_x

def count_row_differences_manual(X,Y):

    num_unique_rows = np.unique(X, axis=0).shape[0]
    print(num_unique_rows)
    x_in_y_counter = 0
    for x_i in range(X.shape[0]):
        if np.any(np.all(X[x_i] == Y,axis=1)):
            x_in_y_counter += 1

    y_in_x_counter = 0
    for y_i in range(Y.shape[0]):
        if np.any(np.all(Y[y_i] == X,axis=1)):
            y_in_x_counter += 1    
    print(x_in_y_counter, y_in_x_counter)
    exit(1)

def analayze_post_weighting(save_path):
    folders = os.listdir(save_path)
    print(folders)
    for counter, f in enumerate(folders):
        main_folder = os.path.join(save_path,f)
        trial_id = int(f.split(":")[1])

        embed_name = "embedding_dict_"+str(trial_id)+"_.pikl"
        with open(os.path.join(main_folder,embed_name), 'rb') as file:
            embed_dict = pickle.load(file)

        data_name = "data_"+str(trial_id)+".npz"
        one_hot_data_name = "one_hot_data_"+str(trial_id)+".npz"

        data = np.load(os.path.join(main_folder,data_name))
        #one_hot_data = np.load(os.path.join(main_folder, one_hot_data_name))
        x = data['x']
        y = data['y']

        weights_name = "weights_"+str(500)+".npz"
        gen_weights = np.load(os.path.join(main_folder,weights_name))['w'].flatten()
        print(gen_weights.shape)
        print(x.shape)
        print(type(x))
        ground_truth_path='./data/censusHouseholdPulse_data/cleaned/ipums_cleaned_combined.csv'
        gt = pd.read_csv(ground_truth_path)
        column_names = gt.columns.tolist()[1:]
        gt = gt.to_numpy()
        gt_probs = gt[:,0]
        gt = gt[:,1:]
        #x_not_in_y, y_not_in_x = count_row_differences_manual(x, gt)

        #print(x_not_in_y)
        #print(y_not_in_x)
        print("HAAAA")

        display_marginals(X=x, Y=gt, 
                          X_probs=gen_weights, 
                          Y_probs=gt_probs,
                          column_names=column_names,
                          dataset_names=['Survey','Census'])
        exit(1)
        
        
        exit(1)
        results_dict = {}
        for i in range(x.shape[0]):
            x_ind = tuple(x[i,:])
            if x_ind not in results_dict:
                results_dict[x_ind] = [gen_weights[i]]
            else:
                results_dict[x_ind].append(gen_weights[i])
        
        for k,v in results_dict.items():
            if len(v) > 1:
                print(v)

def compute_marginals(data, probs=None):
    """
    data  : (N, D) numpy array of integer categorical variables
    probs : (N,) optional probability weights

    Returns:
        list of dicts (length D), each dict maps category -> marginal probability
    """
    N, D = data.shape
    marginals = []

    if probs is None:
        probs = np.ones(N) / N
    else:
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()

    for j in range(D):
        col = data[:, j]
        categories = np.unique(col)
        dist = {}

        for c in categories:
            mask = (col == c)
            dist[int(c)] = float(probs[mask].sum())

        marginals.append(dist)

    return marginals

def display_marginals(X, Y, X_probs=None, Y_probs=None,column_names=None,dataset_names=None):
    """
    Displays side-by-side marginal distributions for X and Y.
    """
    X_marginals = compute_marginals(X, probs=X_probs)
    Y_marginals = compute_marginals(Y, probs=Y_probs)
    X_original = compute_marginals(X, probs=None)

    print()


    D = X.shape[1]

    for j in range(D):
        print(f"\nVariable {column_names[j]}")
        print("-" * 50)

        all_cats = sorted(
            set(X_marginals[j].keys()).union(Y_marginals[j].keys())
        )

        print(f"{'Category':>10} | {'Orig Survey':>15} |{dataset_names[0]:>15} | {dataset_names[1]:>15} | {'difference':>15}")
        print("-" * 50)

        for c in all_cats:
            x_val = X_marginals[j].get(c, 0.0)
            y_val = Y_marginals[j].get(c, 0.0)
            x_val_orig = X_original[j].get(c, 0.0)
            print(f"{c:>10} | {x_val_orig:>15.6f} | {x_val:>15.6f} | {y_val:>15.6f} | {x_val-y_val:>15.6f}")

    return X_marginals, Y_marginals, X_original

def diagnostic_heatmap(runs_dir, target_var_names, diag_var_names, 
                       target_values, interactable=False,
                       filter_by=['lambdaw=0.0']):
    """
    For each tensorboard run in runs_dir, loads target and diagnostic variables
    and produces a 1D/2D/3D heatmap showing how close the target was to its
    benchmark value across combinations of diagnostic variable values.

    Parameters
    ----------
    runs_dir        : str         - path containing tensorboard run subdirectories
    target_var_names: list[str]   - tensorboard scalar names to treat as targets
    diag_var_names  : list[str]   - 1-3 diagnostic variable names (axes of the plot)
    target_values   : list[float] - benchmark value for each target variable (same order)
    """
    import re
    assert 1 <= len(diag_var_names) <= 3, "diag_var_names must have 1–3 entries"
    assert len(target_var_names) == len(target_values), "target_var_names and target_values must match in length"

    # ── 1. Load all runs ──────────────────────────────────────────────────────
    all_points = []   # list of dicts: {diag0: v, diag1: v, diag2: v, 'error': e}
    tfevents_re = re.compile(r'events\.out\.tfevents\.')

    for run_name in os.listdir(runs_dir):
        if filter_by[0] not in run_name:
            continue
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        # find the tfevents file
        event_file = None
        for fname in os.listdir(run_path):
            if tfevents_re.match(fname):
                event_file = os.path.join(run_path, fname)
                break
        if event_file is None:
            continue

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        available = ea.Tags().get('scalars', [])
        # skip runs missing any required variable
        required = target_var_names + diag_var_names
        if not all(v in available for v in required):
            continue

        # load all variables as step→value dicts
        def to_dict(name):
            return {e.step: e.value for e in ea.Scalars(name)}

        diag_dicts   = [to_dict(n) for n in diag_var_names]
        target_dicts = [to_dict(n) for n in target_var_names]

        # find steps present in ALL variables
        common_steps = set(diag_dicts[0].keys())
        for d in diag_dicts[1:] + target_dicts:
            common_steps &= set(d.keys())

        for step in common_steps:
            # mean absolute error across all target variables
            error = np.mean([abs(td[step] - tv) for td, tv in zip(target_dicts, target_values)])
            point = {'error': error}
            for i, dd in enumerate(diag_dicts):
                point[i] = dd[step]
            all_points.append(point)

    if not all_points:
        print("No data found.")
        return

    errors = np.array([p['error'] for p in all_points])
    diag_vals = [np.array([p[i] for p in all_points]) for i in range(len(diag_var_names))]

    # ── 2. Plot ───────────────────────────────────────────────────────────────
    ndim = len(diag_var_names)
    cmap = 'RdYlGn_r'   # green = close to target, red = far

    if ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(diag_vals[0], errors, c=errors, cmap=cmap, alpha=0.6, s=10)
        ax.set_xlabel(diag_var_names[0])
        ax.set_ylabel('|target - benchmark|')
        plt.colorbar(sc, ax=ax, label='error')
        ax.set_title('Target error vs ' + diag_var_names[0])

    elif ndim == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(diag_vals[0], diag_vals[1], c=errors, cmap=cmap, alpha=0.6, s=10)
        ax.set_xlabel(diag_var_names[0])
        ax.set_ylabel(diag_var_names[1])
        plt.colorbar(sc, ax=ax, label='|target - benchmark|')
        ax.set_title('Target error heatmap')

    else:  # ndim == 3
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(diag_vals[0], diag_vals[1], diag_vals[2],
                        c=errors, cmap=cmap, alpha=0.6, s=10)
        ax.set_xlabel(diag_var_names[0])
        ax.set_ylabel(diag_var_names[1])
        ax.set_zlabel(diag_var_names[2])
        fig.colorbar(sc, ax=ax, label='|target - benchmark|', shrink=0.5)
        ax.set_title('Target error heatmap (3D)')

    if interactable:
        import plotly.graph_objects as go
        target_str = ', '.join(f'{n}={v}' for n, v in zip(target_var_names, target_values))
        marker = dict(color=errors, colorscale='RdYlGn_r', size=3, showscale=True,
                      colorbar=dict(title='|target - benchmark|'))
        if ndim == 1:
            trace = go.Scatter(x=diag_vals[0], y=errors, mode='markers', marker=marker)
            layout = go.Layout(xaxis_title=diag_var_names[0], yaxis_title='|target - benchmark|',
                               title=f'Benchmarks: {target_str}')
            go.Figure(data=[trace], layout=layout).show()
        elif ndim == 2:
            trace = go.Scatter(x=diag_vals[0], y=diag_vals[1], mode='markers', marker=marker)
            layout = go.Layout(xaxis_title=diag_var_names[0], yaxis_title=diag_var_names[1],
                               title=f'Benchmarks: {target_str}')
            go.Figure(data=[trace], layout=layout).show()
        else:
            trace = go.Scatter3d(x=diag_vals[0], y=diag_vals[1], z=diag_vals[2],
                                 mode='markers', marker=marker)
            layout = go.Layout(scene=dict(xaxis_title=diag_var_names[0],
                                          yaxis_title=diag_var_names[1],
                                          zaxis_title=diag_var_names[2]),
                               title=f'Benchmarks: {target_str}')
            go.Figure(data=[trace], layout=layout).show()
    else:
        target_str = ', '.join(f'{n}={v}' for n, v in zip(target_var_names, target_values))
        fig.suptitle(f'Benchmarks: {target_str}', fontsize=9)
        plt.tight_layout()
        plt.savefig('./testheatmap.png')

def load_autotune_dir(autotune_dir):
    """
    For each week subdirectory in autotune_dir, load all TensorBoard runs
    under <subdir>/runs/ and return their scalar traces.

    Returns
    -------
    results : dict
        { week_str: { var_name: list[list[float]] } }
        Each inner list is one run; each run is a list of scalar values in step order.
    """
    var_names = ["tvd", "jsd", "RECVDVACC prediction", "Gen Entropy", "Truth-Fake scores"]
    tfevents_re = re.compile(r'events\.out\.tfevents\.')
    week_re = re.compile(r'week(\d+)')

    results = {}

    for subdir in sorted(os.listdir(autotune_dir)):
        m = week_re.search(subdir)
        if m is None:
            continue
        week = m.group(1)
        runs_path = os.path.join(autotune_dir, subdir, "runs")
        if not os.path.isdir(runs_path):
            continue

        week_data = {v: [] for v in var_names}

        for run_name in sorted(os.listdir(runs_path)):
            run_path = os.path.join(runs_path, run_name)
            if not os.path.isdir(run_path):
                continue

            event_file = None
            for fname in os.listdir(run_path):
                if tfevents_re.match(fname):
                    event_file = os.path.join(run_path, fname)
                    break
            if event_file is None:
                continue

            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            available = ea.Tags().get("scalars", [])

            for vname in var_names:
                if vname not in available:
                    week_data[vname].append([])
                    continue
                week_data[vname].append([e.value for e in ea.Scalars(vname)])

        results[week] = {v: np.array(week_data[v]) for v in var_names}

    return results

def summarize_starting_predictions_by_combo(runs_path, num_runs=50):
    '''
    Groups tensorboard run folders in runs_path by the target-variable
    combination baked into each folder's name (the '||labels:[...]' suffix
    experiments.py appends to the SummaryWriter comment), then for each
    combination reports, across all runs sharing that combination:
      - mean/std of the additional target variable's '<VAR> prediction'
        scalar at its first logged step
      - mean/std of each run's own mean over its last third of time steps
      - the same last-third-mean aggregation, but for the baseline
        'RECVDVACC prediction' scalar

    Returns
    -------
    dict: {combo_label_str: {var_name: {'var', 'start_mean', 'start_std',
                                         'end_mean', 'end_std',
                                         'recvdvacc_end_mean', 'recvdvacc_end_std',
                                         'tvd_end_mean', 'tvd_end_std',
                                         'jsd_end_mean', 'jsd_end_std', 'n'}}}
    '''
    def last_third_means(preds):
        runs = [run for run in preds if len(run) > 0]
        return np.array([np.mean(run[-max(1, len(run)//3):]) for run in runs])

    label_re = re.compile(r"labels:(\[[^\]]*\])")
    combo_to_names = {}
    for name in os.listdir(runs_path):
        if not os.path.isdir(os.path.join(runs_path, name)):
            continue
        match = label_re.search(name)
        if match is None:
            continue
        combo_to_names.setdefault(match.group(1), []).append(name)

    results = {}
    for combo_str, names in combo_to_names.items():
        label_names = ast.literal_eval(combo_str)
        extra_vars = [l for l in label_names if l != 'RECVDVACC']
        if not extra_vars:
            extra_vars = label_names
        recvdvacc_tag = "RECVDVACC prediction"
        tvd_tag = "tvd"
        jsd_tag = "jsd"
        var_tags = [f"{v} prediction" for v in extra_vars]

        out, _ = load_runs_as_numpy(runs_path, var_tags + [recvdvacc_tag, tvd_tag, jsd_tag], filter_by=[combo_str],
                                     num_runs=max(len(names), num_runs))
        recvdvacc_ending_means = last_third_means(out[recvdvacc_tag])
        tvd_ending_means = last_third_means(out[tvd_tag])
        jsd_ending_means = last_third_means(out[jsd_tag])

        combo_results = {}
        print()
        print(f"{combo_str}:")
        for var_name, tag in zip(extra_vars, var_tags):
            preds = out[tag]
            runs = [run for run in preds if len(run) > 0]
            starting_values = np.array([run[0] for run in runs])
            if starting_values.size == 0:
                continue
            ending_means = last_third_means(preds)

            combo_results[var_name] = {
                'var': var_name,
                'start_mean': float(np.mean(starting_values)),
                'start_std': float(np.std(starting_values)),
                'end_mean': float(np.mean(ending_means)),
                'end_std': float(np.std(ending_means)),
                'recvdvacc_end_mean': float(np.mean(recvdvacc_ending_means)),
                'recvdvacc_end_std': float(np.std(recvdvacc_ending_means)),
                'tvd_end_mean': float(np.mean(tvd_ending_means)),
                'tvd_end_std': float(np.std(tvd_ending_means)),
                'jsd_end_mean': float(np.mean(jsd_ending_means)),
                'jsd_end_std': float(np.std(jsd_ending_means)),
                'n': int(starting_values.shape[0]),
            }
            r = combo_results[var_name]
            print(f"  {var_name}:")
            print(f"    start: mean={r['start_mean']:.4f}, std={r['start_std']:.4f}")
            print(f"    end:   mean={r['end_mean']:.4f}, std={r['end_std']:.4f}")
            print(f"    n={r['n']}")
        if combo_results:
            print(f"  RECVDVACC end: mean={np.mean(recvdvacc_ending_means):.4f}, std={np.std(recvdvacc_ending_means):.4f}")
            print(f"  tvd (non-witheld JSD) end: mean={np.mean(tvd_ending_means):.4f}, std={np.std(tvd_ending_means):.4f}")
            print(f"  jsd end: mean={np.mean(jsd_ending_means):.4f}, std={np.std(jsd_ending_means):.4f}")
            results[combo_str] = combo_results

    return results

def row_counts_of_smallest_k(arr, k):
    flat_idx = np.argpartition(arr.ravel(), k)[:k]
    rows, cols = np.unravel_index(flat_idx, arr.shape)
    unique, counts = np.unique(rows, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist())), rows, cols

def row_with_smallest_avg(arr, k, p):
    if p is None:
        p = arr.shape[1]
    row_means = arr[:, k:p].mean(axis=1)
    return row_means

def row_with_most_negative_slope(arr, k, p):
    if p is None:
        p = arr.shape[1]
    segment = arr[:, k:p]
    x = np.arange(segment.shape[1])
    slopes = np.array([np.polyfit(x, segment[i], 1)[0] for i in range(arr.shape[0])])
    return np.argmin(slopes)

def save_pickle(runs_path):
    all_outs = {}
    target_var_name = 'RECVDVACC prediction'
    load_filters = ['Week=23', 'Week=24', 'Week=25',
                    'Week=26', 'Week=27', 'Week=28', 'Week=29']

    for filter_by in load_filters:
        num_runs = 10
        out, event_files = load_runs_as_numpy(
            runs_path,
            [target_var_name, 'GenLoss', 'Gen Entropy',
             'tvd', 'Warmup', 'Bias score', 'Gradient Penalty', 'jsd',
             'Truth - Fake scores'],
            filter_by=[filter_by],
            num_runs=num_runs,
        )
        print(out['jsd'].shape)
        all_outs[filter_by] = out
    
    with open('./analysis_data.pikl', 'wb') as file:
        pickle.dump(all_outs, file)
    
def new_stopping():
    target_var_name = 'RECVDVACC prediction'
    with open('./analysis_data.pikl', 'rb') as file:
        all_outs = pickle.load(file)
    
    for fb in all_outs.keys():
        out = all_outs[fb]
        if len(out['RECVDVACC prediction']) == 0:
            continue
        #dictt, row, col = row_counts_of_smallest_k(out['jsd'], k=20)
        jsd = row_with_smallest_avg(out['jsd'], k=200, p=250)
        tvd = row_with_smallest_avg(out['tvd'], k=250, p=None)
        tf = row_with_smallest_avg(out['Truth - Fake scores'], k=400, p=None)
        combined = normalize_matrix(jsd) + normalize_matrix(tvd) +\
              normalize_matrix(tf)
        summ = jsd
        print(fb)
        print("raw: ", summ, np.argmin(summ))
        print("normalized: ", combined, np.argmin(combined))
        tvar = np.mean(out[target_var_name][:,-400:],axis=1)
        print("tvar: ", tvar)
        jsdd = np.mean(out['jsd'][:,-400:],axis=1)
        print(jsdd, np.mean(jsdd))
        stdd = np.std(out['jsd'][:,-400:],axis=1)
        print(stdd, np.mean(stdd))
        print("")


def normalize_matrix(m):
    mn, mx = m.min(), m.max()
    return (m - mn) / (mx - mn) if mx > mn else np.zeros_like(m)

if __name__ == "__main__":
    runs_path = "runs_inctot_race_witheld_FBDelphidriven/"
    summarize_starting_predictions_by_combo(runs_path)
    if False: #analyzing target prediction runs aggregation
        target_var_tb_name = "RECVDVACC prediction"
        metric_tb_name = "jsd" #"Gen Entropy"

        for i in ['2.5']:
            indices = entropy_pred_relationship(runs_path,
                                            filter_by=["Week=25", "lambdaJSD:"+i], 
                                                num_runs=15,
                                                target_var_name=target_var_tb_name,
                                                metric_tb_name=metric_tb_name,
                                                K=100)
    elif False: #analyze post weighting
        save_path = "./saves_week29/"
        analayze_post_weighting(save_path)
    
    elif False: #headmap manual run
        diagnostic_heatmap(runs_dir = './runs/', 
                           target_var_names=['RECVDVACC prediction'], 
                           diag_var_names=['tvd','jsd','Gen Entropy'], 
                           target_values=[0.6])
    
    elif False:
        all_results = load_autotune_dir(autotune_dir="./hyperparam_tuning/autotune_history/autoTuneDir_HHP_Phase4Only")
    elif False:
        #new_stopping()
        save_pickle(runs_path)
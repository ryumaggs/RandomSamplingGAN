#Global imports
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

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
    for i_name, vname in enumerate(var_names):
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

def entropy_pred_relationship(runs_path, filter_by, num_runs):
    out, event_files = load_runs_as_numpy(runs_path,  
                             ['target prediction', 'GenLoss', 'Gen Entropy', 
                              'tvd', 'Warmup', 'Bias score', 'Gradient Penalty'],
                             filter_by=filter_by,
                             num_runs=num_runs)
    vpt = out['target prediction']
    gloss = out['GenLoss']
    entropy = out['Gen Entropy']
    warmup = out['Warmup']
    bscore = out['Bias score']
    gp = out['Gradient Penalty']
    indices_in_range = np.where((entropy[0] >= 0.75) & (entropy[0] <= 0.8))[0]
    print(indices_in_range)
    #t_f_scores = out['Truth - Fake scores']
    tvd = out['tvd']
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    order = np.arange(vpt.shape[0])
    start_point = 0
    print(vpt.shape)
    end_point = len(entropy[0])
    indices = np.arange(len(entropy[0,start_point:end_point]))
    labels = ['entropy', 'truth-fake', 'tvd']
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
        print(minidx, np.mean(vpt[i,minidx-2:minidx+2]))
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

    #data = np.load("./saves/data.npz")
    #weights = np.load("./saves/weight_history.npz")

    #labels = data['y']
    #weights = weights['w']
    #pws = (weights @ labels).T
    #print("diff: ", np.linalg.norm(pws-vpt))

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

def process_all_weeks_dict(all_outs,
                           save_dir,
                           save=False,):
    for key in all_outs:
        if key == 'ALL':
            continue
        entropy = all_outs[key]['Gen Entropy']
        vpt = all_outs[key]['target prediction']
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
            
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from itertools import combinations
#analyzing consistency over different types of randomness
#plotting events from saved run logs
from tensorboard.backend.event_processing import event_accumulator
import os

def load_runs_as_numpy(runs_path, var_names, filter_by, num_runs=20):
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
    for i, event_file in enumerate(event_files):
        for i_name, vname in enumerate(var_names):
            
            # Load event accumulator
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            if not vname in ea.Tags()["scalars"]:
                continue
            if vname not in all_data:
                all_data[vname] = []
            prediction_events = ea.Scalars(vname)
            cur_pred = []
            for event in prediction_events:
                cur_pred.append(event.value)
            all_data[vname].append(cur_pred)
    
    for vname in all_data:
        all_data[vname] = np.array(all_data[vname])
    return all_data, event_files


def save_final_median(path_to_saves,):

    runs_path = path_to_saves+"/runs/"
    all_outs = {}
    out, _ = load_runs_as_numpy(runs_path,  
                             ['target prediction', 'GenLoss', 'Gen Entropy', 
                              'tvd', 'Warmup', 'Bias score', 'Gradient Penalty'],
                             filter_by=[""],
                             num_runs=20)
    entropy = out['Gen Entropy']
    vpt = out['target prediction']
    all_min_entropy_idx = np.argmin(entropy,axis=1)
    all_min_entropy_preds = []
    for i, amei in enumerate(all_min_entropy_idx):
        all_min_entropy_preds.append(vpt[i,amei])
    all_min_entropy_preds = np.array(all_min_entropy_preds)
    sorted_min_entropy_preds = np.sort(all_min_entropy_preds)

    #isolate out the median and which trial it came from
    median = sorted_min_entropy_preds[(sorted_min_entropy_preds.shape[0]-1)//2]
    selected_trial = np.where(all_min_entropy_preds == median)[0].item()

    #load in the weights and pick out the corresponding weight
    data = np.load(path_to_saves+"/data_"+str(selected_trial)+".npz")
    weights = np.load(path_to_saves+"/weight_history_"+str(selected_trial)+".npz")
    datum = data['x']
    labels = data['y']
    weights = weights['w']
    pws = ((weights @ labels).T.flatten())
    #print(datum.shape, labels.shape)
    closest_time_step = np.where(np.isclose(pws,median))[0].item()
    selected_weights = weights[closest_time_step,:]
    np.savez(path_to_saves+"/chosen_weight.npz", w=selected_weights)
    np.savez(path_to_saves+"/chosen_data.npz", x = datum, y = labels)
    np.savez(path_to_saves+"/chosen_weight_history.npz", w = weights)
    print("Median estimate:", median)
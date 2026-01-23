import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from util import embed_data

def randomly_select_valid_points(X,
                                 D,
                                 K):
    '''
    X - dataset - numpy array
    D - condition dictionary - dict[int] = str - key = var index, val = bucket index of var
    K - int - number of points to select
    '''
    # Start with a mask that selects all rows
    mask = np.ones(len(X), dtype=bool)

    if D is not None:
        # Apply each condition
        for feature_idx, category_id in D.items():
            mask &= (X[:, feature_idx] == category_id)

    # Filter rows
    matching_rows = np.where(mask)[0]

    if len(matching_rows) == 0:
        print("No matching rows found.")
        return np.empty((0, X.shape[1]))

    # Randomly sample up to K rows
    if K != np.inf:
        selected_indices = np.random.choice(matching_rows, size=min(K, len(matching_rows)), replace=False)
    else:
        selected_indices = matching_rows
    print(len(selected_indices))
    return selected_indices

def compute_equispaced_inputs_IG(baseline_input,actual_input,num_steps,unsqueeze=False):
    '''
    baseline_input - the mean , shape 1 x N
    acutal_input - data set, shape M x N
    num_steps - int - number of equispaced steps between baseline and actual

    Finds num_steps equipspaced inputs between the baseline and actual input 
    '''
    all_inputs = []
    # Compute difference from mu to each row in D
    diff = actual_input - baseline_input  # 
    # Generate interpolation coefficients between 0 and 1
    alphas = torch.linspace(0, 1, steps=num_steps)  # 0 -> mu, 1 -> D

    # Create list of interpolated matrices
    for alpha in alphas:
        # Linear interpolation for all rows
        interp = baseline_input + alpha * diff
        if unsqueeze:
            all_inputs.append(interp.unsqueeze(0))
        else:
            all_inputs.append(interp)
    return all_inputs

def compute_IG_jacobian(all_inputs,gan,network='generator'):
    '''
    computes full jacobian with respect to input
    '''
    all_grads = []
    for ai in all_inputs:
        X_for_grad = ai.detach().clone().requires_grad_(True)
        # Forward pass through G then D
        sds, selected_indices, weights = gan.generator(X_for_grad)
        #score = gan.discriminator(sds).sum()  # scalar score

        # Compute gradient of score w.r.t. generator input
        jac = []
        for i in range(weights.numel()):
            grad_input = torch.autograd.grad(
                outputs=weights.view(-1)[i],
                inputs=X_for_grad,
                retain_graph=True,
                create_graph=False
            )[0].detach()  # shape: (2500, 7)
            jac.append(torch.sqrt(grad_input**2).sum().cpu().item())
        all_grads.append(np.mean(jac))
    return np.mean(all_grads)

def compute_IG_weights(all_inputs,
                       baseline_input,
                       actual_input,
                       gan,
                       unscaled_dataset,):
    '''
    all_inputs - list[torch.tensor] - returned by "compute equispaced_inputs_IG"
    network - str - 'generator' or 'critic'
    num_steps - int - number of steps between baseline and actual to compute gradient

    A few steps to take into consideration:
    1. How many points are we computing the int grad for, cannot do all, lets say K
    2. How to identify K points? we have the unscaled data set
    3. Need to convert the multiplying vector into One_hot space appropriately
    4. Can we answer multiple questions with these limited K points
        - for instance we randomly pick K points and then just track those K points over time
    5. Or do we, at each time step, pick a new random K points and whatever points we get is whatever we get

    Tracking same points over time:
        Pros: can see track these points over time to see how answer changes
        Cons: For a specific "question" need to redo the experiment. which, one iteration ~3 min
    
    Each time step pick random point and keep track of questions:
        pros: can see wide range of questions/answers over time
        cons: do we really care about over time? or just a specific time. 


    '''
    #tD = {}
    #tD[3] = 0
    tD = None
    indexes = randomly_select_valid_points(X=unscaled_dataset,D=tD,K=np.inf)
    all_grads = [[] for _ in range(len(indexes))]
    integrated_grads = [None for _ in range(len(indexes))]
    for ai in tqdm(all_inputs):
        X_for_grad = ai.detach().clone().requires_grad_(True)

        # Forward pass through G then D
        _, _, logits = gan.generator(X_for_grad)
        softmaxed_logits = torch.nn.functional.softmax(logits,dim=1)

        # Compute gradient of score w.r.t. generator input
        for i, ind in enumerate(indexes):
            grad_input = torch.autograd.grad(
                outputs=softmaxed_logits[0,ind],
                inputs=X_for_grad,
                retain_graph=True,
                create_graph=False,
            )[0].detach()  # shape: (2500, 7)
            all_grads[i].append(grad_input.cpu().numpy())
    
    for i in range(len(indexes)):
        all_grads[i] = np.concatenate(all_grads[i],axis=0) 
        all_grads[i] = np.mean(all_grads[i],axis=0) ##aggregates over all_inputs
        integrated_grads[i] = (actual_input-baseline_input).cpu().numpy() * all_grads[i] #scales by diff(actual,baseline)
        integrated_grads[i] = integrated_grads[i][indexes[i],:]
    
    return integrated_grads
    
def compute_IG_score(all_inputs,baseline_input,actual_input,gan,network='generator'):
    '''
    all_inputs - list[torch.tensor] - returned by "compute equispaced_inputs_IG"
    network - str - 'generator' or 'critic'
    num_steps - int - number of steps between baseline and actual to compute gradient

    1. iterates over each input in the list and computes the average gradient associated with the input
    for the appropriate network type

    '''
    if network == 'generator':
        all_grads = []
        for ai in all_inputs:
            X_for_grad = ai.detach().clone().requires_grad_(True)

            # Forward pass through G then D
            sds, selected_indices, weights = gan.generator(X_for_grad)
            score = gan.discriminator(sds).sum()  # scalar score

            # Compute gradient of score w.r.t. generator input
            grad_input = torch.autograd.grad(
                outputs=score,
                inputs=X_for_grad,
                retain_graph=False,
                create_graph=False
            )[0].detach()  # shape: (2500, 7)

            all_grads.append(grad_input.cpu().numpy())
        all_grads = np.concatenate(all_grads,axis=0)
        all_grads = np.mean(all_grads,axis=0)
        integrated_grads = (actual_input-baseline_input).cpu().numpy() * all_grads
        return integrated_grads
    
    elif network == 'critic':
        '''
        if critic, assumes that each input of all inputs is
        the selected subset and should therefore be passed directly to the critic
        '''
        all_grads = []
        for ai in all_inputs:
            X_for_grad = ai.detach().clone().requires_grad_(True)
            score = gan.discriminator(X_for_grad).sum()  # scalar score

            # Compute gradient of score w.r.t. generator input
            grad_input = torch.autograd.grad(
                outputs=score,
                inputs=X_for_grad,
                retain_graph=False,
                create_graph=False
            )[0].detach()  # shape: (2500, 7)
            all_grads.append(grad_input.cpu().numpy())
        all_grads = np.concatenate(all_grads,axis=0)
        all_grads = np.mean(all_grads,axis=0)
        integrated_grads = (actual_input-baseline_input).cpu().numpy() * all_grads
        return integrated_grads
    else:
        raise NotImplementedError
    
def compute_IG_depricated101325(all_inputs,baseline_input,actual_input,gan,network='generator'):
    '''
    all_inputs - list[torch.tensor] - returned by "compute equispaced_inputs_IG"
    network - str - 'generator' or 'critic'
    num_steps - int - number of steps between baseline and actual to compute gradient

    1. iterates over each input in the list and computes the average gradient associated with the input
    for the appropriate network type

    '''
    if network == 'generator':
        all_grads = []
        for ai in all_inputs:
            X_for_grad = ai.detach().clone().requires_grad_(True)

            # Forward pass through G then D
            sds, selected_indices, weights = gan.generator(X_for_grad)
            score = gan.discriminator(sds).sum()  # scalar score

            # Compute gradient of score w.r.t. generator input
            grad_input = torch.autograd.grad(
                outputs=score,
                inputs=X_for_grad,
                retain_graph=False,
                create_graph=False
            )[0].detach()  # shape: (2500, 7)

            all_grads.append(grad_input.cpu().numpy())
        all_grads = np.concatenate(all_grads,axis=0)
        return np.mean(all_grads,axis=0)
    
    elif network == 'critic':
        '''
        if critic, assumes that each input of all inputs is
        the selected subset and should therefore be passed directly to the critic
        '''
        all_grads = []
        for ai in all_inputs:
            X_for_grad = ai.detach().clone().requires_grad_(True)
            score = gan.discriminator(X_for_grad).sum()  # scalar score

            # Compute gradient of score w.r.t. generator input
            grad_input = torch.autograd.grad(
                outputs=score,
                inputs=X_for_grad,
                retain_graph=False,
                create_graph=False
            )[0].detach()  # shape: (2500, 7)
            all_grads.append(grad_input.cpu().numpy())
        all_grads = np.concatenate(all_grads,axis=0)
        return np.mean(all_grads,axis=0)
    else:
        raise NotImplementedError

def process_grad(cgrad,embedding_dict):
    '''
    cgrad - where each row is the gradients associated with all features of a data point
    embedding_dict - dictionary where keys are true feature index and values contain onehot/embedding information
    
    cgrad should be the gradient for a single timestep
    the "current gradient"
    '''
    sep_grads = {}
    cur_index = 0
    visited_columns = np.zeros(cgrad.shape[1])
    for k,e in embedding_dict.items():
        #sep_grads[k] = np.linalg.norm(cgrad[:,cur_index:cur_index+e[2]],ord=2)
        sep_grads[k] = np.mean(cgrad[:,cur_index:cur_index+e[2]])
        visited_columns[cur_index:cur_index+e[2]] = 1
        cur_index += e[2]
    return sep_grads

def find_all_nn_weights_NN(tree,
                        dataset,
                        weights,
                        rows,
                        var_index,
                        new_var_value):
    new_points_avg_weight = []
    for r in rows:
        point = np.copy(dataset[r])
        point[var_index] = new_var_value
        dists, idx = tree.query(point, k=5)
        new_points_avg_weight.append(np.mean(weights[idx]))
    return new_points_avg_weight

def analyze_question_NN(dataset,
                     tree,
                     weights,
                     question):
    '''
    dataset - loaded np array
    labels - loaded np array
    weights - loaded np array
    question - dict[intA] = [intB,intC] | intA = variable index,
    int B = starting category, int C = new category
    '''
    var_index = list(question.keys())[0]
    var_start = list(question.values())[0][0]
    var_end = list(question.values())[0][1]
    rows = np.where(dataset[:, var_index] == var_start)[0]

    
    new_points_avg_weight = find_all_nn_weights_NN(tree,
                        dataset,
                        weights,
                        rows,
                        var_index,
                        var_end)
    
    #print(new_points_avg_weight)
    #print(weights[rows])

    avg_diff = np.mean(new_points_avg_weight-weights[rows])
    return avg_diff

def compute_confusion_matrix_NN(dataset,
                            labels,
                            weights,
                            embed_dict,):
    avg_weight = np.mean(weights)
    all_matrices = []
    tree = cKDTree(dataset)
    for var_id in embed_dict:
        num_categories = embed_dict[var_id][2]
        c_matrix = np.zeros((num_categories,num_categories))
        for start_val in range(num_categories):
            for new_val in range(num_categories):
                if new_val == start_val:
                    continue
                c_question = {var_id:[start_val,new_val]}
                avg_diff = analyze_question_NN(dataset, tree, weights, c_question)
                c_matrix[start_val,new_val] = avg_diff
        all_matrices.append(c_matrix)
    return all_matrices

def find_all_nn_logits_GEN_singular(generator,
                        dataset,
                        embeded_dataset,
                        embed_dict,
                        rows,
                        var_index,
                        new_var_value):
    '''
    dataset: 2d numpy array, N x 7
    embeded_dataset: 3d torch tensor, N x 33
    '''

    new_points_avg_logit = []
    #embeded_dataset = embeded_dataset.to(torch.device('cuda:0'))
    for r in rows:
        new_point = torch.tensor(np.copy(dataset[r])).unsqueeze(0)
        original_point = torch.clone(embeded_dataset[0,r,:])
        new_point[0,var_index] = new_var_value
        #embed new point
        embeded_new_point = embed_data(None,embed_dict,new_point,overrite_start_idx=0)
        #point embededdataset to it
        embeded_dataset[0,r,:] = embeded_new_point.to(torch.device('cuda:0'))
        #run through generator
        new_logits = generator.get_logits(embeded_dataset) #1 x N, 2d shape
        new_points_avg_logit.append(new_logits[0,r])
        #put original point back
        embeded_dataset[0,r,:] = original_point
    return np.array(new_points_avg_logit)

def find_all_nn_logits_GEN(generator,
                           dataset,
                           embeded_dataset,
                           embed_dict,
                           rows,
                           var_index,
                           new_var_value,
                           max_batch_size=10):
    """
    dataset:          numpy array  [N, 7]
    embeded_dataset:  torch tensor [1, N, 33]
    rows:             indices to modify
    """

    N = embeded_dataset.shape[1]  # number of datapoints
    D = embeded_dataset.shape[2]  # embed dim

    all_logits = []

    # Process rows in batches of ≤ max_batch_size
    for start in range(0, len(rows), max_batch_size):
        batch_rows = rows[start:start + max_batch_size]
        R = len(batch_rows)

        # Create batch inputs: [R, N, D]
        batch_dataset = embeded_dataset.repeat(R, 1, 1).clone().to(next(generator.phi.parameters()).device)

        # Modify each row in its corresponding batch index
        for i, r in enumerate(batch_rows):

            # Copy original point and alter one variable
            new_point = torch.tensor(dataset[r], device=next(generator.phi.parameters()).device).unsqueeze(0)
            new_point[0, var_index] = new_var_value

            # Re-embed the modified point
            embeded_new_point = embed_data(None, embed_dict, new_point, overrite_start_idx=0)
            embeded_new_point = embeded_new_point.squeeze(0)  # shape [D]

            # Insert modified point into batch i
            batch_dataset[i, r, :] = embeded_new_point

        # Forward pass (single call for whole batch)
        # Output: [R, N]
        new_logits = generator.get_logits(batch_dataset)

        # Extract logits for the modified rows
        # We pull (i, batch_rows[i])
        for i, r in enumerate(batch_rows):
            all_logits.append(new_logits[i, r].item())

    return np.array(all_logits)

def analyze_question_GEN(dataset,
                         embeded_dataset,
                         generator,
                         embed_dict,
                        logits,
                        question,
                        weights,):
    '''
    dataset - loaded np array
    labels - loaded np array
    logits - loaded np array, 2d in shape
    question - dict[intA] = [intB,intC] | intA = variable index,
    int B = starting category, int C = new category
    '''
    var_index = list(question.keys())[0]
    var_start = list(question.values())[0][0]
    var_end = list(question.values())[0][1]
    rows = np.where(dataset[:, var_index] == var_start)[0]
    new_points_logits = find_all_nn_logits_GEN(generator,
                        dataset,
                        embeded_dataset,
                        embed_dict,
                        rows,
                        var_index,
                        var_end)
    diff_arr = new_points_logits-logits[0,rows,0]
    uniform_avg_diff = np.mean(diff_arr)
    weighted_avg_diff = np.sum(weights[rows] * diff_arr)
    return uniform_avg_diff, weighted_avg_diff

def compute_confusion_matrix_GEN(dataset,
                                 embeded_dataset,
                            generator,
                            logits,
                            embed_dict,
                            weights,):
    all_matrices_uniform = []
    all_matrices_weighted = []
    for var_id in embed_dict:
        print("------------------------")
        print("var idx: ", var_id, " out of ", len(embed_dict))
        num_categories = embed_dict[var_id][2]
        c_matrix_uniform = np.zeros((num_categories,num_categories))
        c_matrix_weighted = np.zeros((num_categories,num_categories))
        for start_val in range(num_categories):
            print("start_val: ", start_val)
            for new_val in range(num_categories):
                if new_val == start_val:
                    continue
                c_question = {var_id:[start_val,new_val]}
                avg_diff_uniform, avg_diff_weighted = analyze_question_GEN(dataset,
                                                embeded_dataset,
                                                generator,
                                                embed_dict,
                                                logits,
                                                c_question,
                                                weights,)
                c_matrix_uniform[start_val,new_val] = avg_diff_uniform
                c_matrix_weighted[start_val,new_val] = avg_diff_weighted
        print("")
        print("-------------------------")
        all_matrices_uniform.append(c_matrix_uniform)
        all_matrices_weighted.append(c_matrix_weighted)
    return all_matrices_uniform, all_matrices_weighted


if __name__ == '__main__':
    #Logit difference Computation 
    from Generator import DeepSetNet
    from util import embed_data
    from util import set_seed
    import pickle

    '''
    For a categorical feature like SEX with levels {Male, Female}:

    Take each unit i whose SEX value is Male.

    Form a counterfactual dataset 𝑋′where only that unit’s SEX is flipped to Female.

    Run the generator on 𝑋′ and on the original 𝑋.

    Compute the difference in the generator’s output for that same unit.

    Average this difference across all male units.
    → This gives the Male → Female entry in the heatmap.

    Swap roles and flip Female → Male to populate the other entry.

    This logic is correct.
    '''
    SAME_DATA_GT = False
    SAME_DATA_BIAS = False
    SAME_DATA_SEEN = False
    SAME_NETWORK_INIT = False
    device = torch.device('cuda:0')
    #load generator, load dataset, load weights
    folder = "./saves_week29/"
    data_name = "data_0.npz"
    one_hot_data_name = "one_hot_data_0.npz"
    weights_name = "weights_0_.npz"
    embed_name = "embedding_dict_0_.pikl"


    data = np.load(folder + data_name)
    one_hot_data = np.load(folder + one_hot_data_name)
    x = data['x']
    y = data['y']
    one_hot_x = torch.tensor(one_hot_data['x'],dtype=torch.float32)
    with open(folder + embed_name, 'rb') as file:
        embed_dict = pickle.load(file)
    weights = np.load(folder+weights_name)['w'].flatten()

    checkpoint = torch.load(folder + "/generator_checkpoint.pt", map_location="cpu")
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]
    fixed_seed = np.random.randint(1e6)
    all_rngs = []
    print('SETTING SEED: ', fixed_seed)
    for _ in range(1):
        all_rngs.append(set_seed(fixed_seed,
                                device,
                            data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                            data_gen =SAME_DATA_SEEN,
                            network_init=SAME_NETWORK_INIT,))
        
    generator = DeepSetNet(rngs=all_rngs[0],
                            num_features = config["num_features"],
                            layers = config["layers"],
                            sample_size = config["sample_size"],
                            dropout = config["dropout"],
                            batch_size = config["batch_size"],
                            temperature = config["temperature"],
                            embedding_dict=config['embedding_dict']).to(device)
    missing, unexpected = generator.load_state_dict(checkpoint["state_dict"], strict=False)
    generator.set_eval()

    cur_logits = generator.get_logits(one_hot_x.to(device))

    all_conf_matrices_uniform, all_conf_matrices_weighted = compute_confusion_matrix_GEN(x,
                                                    one_hot_x,
                                                generator,
                                                cur_logits,
                                                embed_dict,
                                                weights,)
    
    with open(folder+"/temp_conf_uniform_matrix.pikl", 'wb') as file:
        pickle.dump(all_conf_matrices_uniform, file)
    
    with open(folder+"/temp_conf_weighted_matrix.pikl", 'wb') as file:
        pickle.dump(all_conf_matrices_weighted, file)
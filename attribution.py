import torch
import numpy as np

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

def compute_IG(all_inputs,gan,network='generator'):
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

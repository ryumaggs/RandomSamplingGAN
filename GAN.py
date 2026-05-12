import torch
import torch.nn as nn
import numpy as np
import random
from Generator import onesGen, dataGen, weightsGen, DeepSetNet, DeepSetComplexNet
from Discriminator import Discriminator, DeepSetCritic, DeepSetCriticComplex
import sys, os
sys.path.append(os.path.dirname(__file__))
from util import dict2vector, get_avg_grad_per_layer, compute_data_shape, embed_data
from tqdm import tqdm
import pytorch_warmup as warmup
import torch.autograd as autograd
from scipy.spatial.distance import jensenshannon
import copy
import pandas as pd
import pickle
from attribution.attribution import *

class GAN():
    def __init__(self,
                 rngs,
                 dataset,
                 generator_type,
                 discriminator_type,
                gen_learning_rate,
                disc_learning_rate,
                batch_size,
                truth_sample_size,
                gen_layers,
                disc_layers,
                bias_sample_size,
                lambda_gp,
                lambda_weights,
                lambda_demo,
                lambda_JSD,
                gen_history_length=10,
                temperature=0.1,
                warmup_length = 1000,
                lambda_regularizer = 0,
                lambda_first_layer = 0,
                generator_dropout = 0.1,
                discriminator_dropout = 0.1,
                KLIEP_downsample = 2500,
                device=torch.device('cuda:0'),
                save_dict={},
                save_dir = "",):

        '''
        subset_size,
                 batch_size,
                 temperature,
                 output_size,
                 device
        '''
        self.entropy_history = []
        self.embedding_dict = dataset.embedding_dict
        self.lambda_first_layer = lambda_first_layer
        self.input_grad_history = []
        self.critic_input_grad_history = []
        self.save_dir = save_dir
        self.save_dict = save_dict
        self.discriminator_type = discriminator_type
        self.disc_learning_rate = disc_learning_rate
        self.disc_layers = disc_layers
        self.discriminator_dropout = discriminator_dropout
        self.sample_size = truth_sample_size
        self.weights_history = []
        self.gen_history_length = gen_history_length
        self.generator = None
        self.discriminator = None
        self.rngs=rngs
        self.start_decay = 100
        self.batch_size = batch_size
        self.device=device
        self.temperature = temperature
        self.generator_type = generator_type
        self.lambda_gp = lambda_gp
        self.lambda_weights=lambda_weights
        self.lambda_demo = lambda_demo
        self.lambda_jsd = lambda_JSD
        self.discriminator_starting_lr = disc_learning_rate
        self.generator_starting_lr = gen_learning_rate
        self.final_dataset_shape = dataset.biased_dataset.shape[2]

        if generator_type == 'standard':
            self.generator = dataGen(rngs=self.rngs,
                                     num_features=self.final_dataset_shape,
                                     layers=gen_layers,
                                     sample_size=bias_sample_size,
                                     dropout=generator_dropout,
                                     temperature=temperature,
                                     embedding_dict=self.embedding_dict).to(self.device)
        elif generator_type == 'deepSet':
            self.generator = DeepSetNet(rngs=self.rngs,
                                        num_features=self.final_dataset_shape,
                                     layers=gen_layers,
                                     sample_size=bias_sample_size,
                                     dropout=generator_dropout,
                                     batch_size = batch_size,
                                     temperature=temperature,
                                     embedding_dict=self.embedding_dict).to(self.device)
        elif generator_type == 'deepSetComplex':
            self.generator = DeepSetComplexNet(rngs=self.rngs,
                                        num_features=self.final_dataset_shape,
                                     layers=gen_layers,
                                     sample_size=bias_sample_size,
                                     dropout=generator_dropout,
                                     batch_size = batch_size,
                                     temperature=temperature,
                                     embedding_dict=self.embedding_dict).to(self.device)
        elif generator_type == 'weights':
            self.generator = weightsGen(survey_dataset=dataset.biased_dataset,
                                        sample_size=bias_sample_size,).to(self.device)
        else:
            raise NotImplementedError
    
        self.generator_config = {
                                    "rng_seed": 0,
                                    "num_features": self.final_dataset_shape,
                                    "layers": gen_layers,
                                    "sample_size": bias_sample_size,
                                    "dropout": generator_dropout,
                                    "batch_size": batch_size,
                                    "temperature": temperature,
                                    "embedding_dict": self.embedding_dict,
                                }
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                    lr=gen_learning_rate,
                                                    betas=(0, 0.9),
                                                    weight_decay=1e-4)
        
        self.reset_discriminator()

        self.loss_function = nn.BCEWithLogitsLoss()
        
        #self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.generator_optimizer, gamma=1)
        #self.generator_warmup_scheduler = warmup.LinearWarmup(self.generator_optimizer, warmup_length)

        
        #self.discriminator_warmup_scheduler = warmup.LinearWarmup(self.discriminator_optimizer, warmup_length)
        
        self.num_categories_per_features = dataset.num_categories_per_features

        self.label_names = dataset.label_names
        self.label_indexes = dataset.label_indexes
        self.num_labels = dataset.num_labels

        self.gt_weights = torch.tensor(dataset.gt_weights)
        self.ground_truth_dataset = dataset.ground_truth_dataset
        self.bias_dataset = dataset.biased_dataset

        self.unscaled_ground_truth = torch.tensor(dataset.unscaled_ground_truth).to(device)
        self.unscaled_biased = torch.tensor(dataset.unscaled_biased).to(device)

        self.ground_truth = dataset.ground_truth
        self.truth_sample_size = truth_sample_size    
        self.data_type = dataset.type

        self.lambda_regularizer = lambda_regularizer
        self.biased_labels = dataset.biased_labels  
        self.ground_truth_demographics = dataset.ground_truth_demographics 

        self.gt_cpu = dataset.gt_cpu
        self.bias_cpu = dataset.bias_cpu

        if KLIEP_downsample == -1:
            self.KLIEP_downsample = -1
        else:
            self.KLIEP_downsample=min(KLIEP_downsample,self.unscaled_biased.shape[0])

        self.original_distribution = None
        if hasattr(dataset, "original_distribution"):
            self.col_indexes = dataset.col_indexes
            self.original_distribution = dataset.original_distribution
            self.var_counts = dataset.var_counts

    def compute_input_gradient_generator(self):
        #generator code
        baseline_input=torch.mean(self.ground_truth_dataset,dim=0).to(self.device)
        all_inputs = compute_equispaced_inputs_IG(baseline_input=baseline_input,
                                        actual_input=self.bias_dataset.squeeze(0),
                                        num_steps=32,
                                        unsqueeze=True)
        mean_grad = compute_IG_weights(all_inputs=all_inputs,
                                         baseline_input=baseline_input,
                                         actual_input=self.bias_dataset.squeeze(0),
                                         gan=self,
                                         unscaled_dataset=self.unscaled_biased,
                                         )
        
        self.input_grad_history.append(mean_grad)

    def compute_input_gradient_disc(self):
        critic_actual_input = None
        with torch.no_grad():
            # Forward pass through G then D
            critic_actual_input, _, _ = self.generator(self.bias_dataset)
        critic_actual_input = critic_actual_input[0,:,:]
        all_inputs_critic = compute_equispaced_inputs_IG(baseline_input=torch.mean(self.ground_truth_dataset,dim=0),
                                        actual_input=critic_actual_input,
                                        num_steps=50,
                                        unsqueeze=True)
        mean_grad_critic = compute_IG_corrected(all_inputs=all_inputs_critic,
                                         baseline_input=torch.mean(self.ground_truth_dataset,dim=0),
                                         actual_input=critic_actual_input,
                                         gan=self,
                                         network='critic')
        self.critic_input_grad_history.append(mean_grad_critic)
    
    def reset_discriminator(self):
        if self.discriminator_type == 'standard':
            self.discriminator = Discriminator(rngs=self.rngs,
                                               num_features=self.final_dataset_shape*self.sample_size,
                                            dropout=self.discriminator_dropout,
                                            layers=self.disc_layers,
                                            embedding_dict=self.embedding_dict).to(self.device)
        elif self.discriminator_type == 'deepSet':
            self.discriminator = DeepSetCritic(rngs=self.rngs,
                                               num_features=self.final_dataset_shape,
                                            dropout=self.discriminator_dropout,
                                        layers=self.disc_layers,
                                        embedding_dict=self.embedding_dict).to(self.device)
        elif self.discriminator_type == 'deepSetComplex':
            self.discriminator = DeepSetCriticComplex(rngs=self.rngs,
                                               num_features=self.final_dataset_shape,
                                            dropout=self.discriminator_dropout,
                                        layers=self.disc_layers,
                                        embedding_dict=self.embedding_dict).to(self.device)
        else:
            raise NotImplementedError

        self.critic_config = {
                            "rng_seed": 0,
                            "num_features": self.final_dataset_shape,
                            "dropout": self.discriminator_dropout,
                            "layers": self.disc_layers,
                            "embedding_dict": self.embedding_dict,
                            }
        
        
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=self.disc_learning_rate,
                                                    betas=(0, 0.9),
                                                    weight_decay=1e-4)
        gamma = (1e-5 / self.disc_learning_rate) ** (1 / 400)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=gamma)
    
    def set_temperature(self, new_temp):
        self.temperature = new_temp

    def generate_data(self,batch_override=None):
        batches = self.batch_size
        if batch_override is not None:
            batches = batch_override
        #assert 1 == 0, "need to figure out what device and type ground_truth_dataset is here. sample WITHOUT replacement"
        index = self.rngs['np'].choice(np.arange(len(self.ground_truth_dataset)),size=self.truth_sample_size*batches,
                                       replace=False)
        sampled_data = torch.clone(self.ground_truth_dataset[index])
        sampled_data = torch.reshape(sampled_data, (batches, self.truth_sample_size, self.ground_truth_dataset.shape[1])).to(self.device)
        #sampled_data = torch.reshape(sampled_data,(sampled_data.shape[0],sampled_data.shape[1]*sampled_data.shape[2]))
        return sampled_data

    def get_reg_loss(self):
        epsilon = 1e-5
        
        weights = self.generator.get_weights_regularizer(self.bias_dataset)
        return torch.square(torch.norm(input=weights,p=2)) * self.bias_dataset.shape[1]
        #return -torch.sum(weights * torch.log(weights + epsilon)) entropy experiments were hard to balance
    
    def get_medium_entropy_loss(self):
        probabilities = self.generator.get_weights_regularizer(self.bias_dataset)

        epsilon = 1e-12
        probs_clamped = probabilities.clamp(min=epsilon)

        # Entropy: H(p) = -sum(p * log(p))
        entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)

        # Normalize entropy by log(K)
        K = probabilities.shape[1]
        max_entropy = torch.log(torch.tensor(K, dtype=entropy.dtype, device=entropy.device))
        normalized_entropy = entropy / max_entropy  # range: [0, 1]
        # Medium entropy target range: [0.3, 0.7]
        lower, upper = 0.5, 0.7
        lower_penalty = torch.relu(lower - normalized_entropy)
        upper_penalty = torch.relu(normalized_entropy - upper)

        reg_loss = lower_penalty**2 + upper_penalty**2
        return reg_loss

    def get_demo_loss(self):
        weights = self.generator.get_weights_regularizer(self.bias_dataset)
        weighted_gt = torch.mean(self.ground_truth_dataset, dim=0)
        #get mean of prediction with weights
        weighted_avg = (self.bias_dataset*weights.T).sum(dim=0)
        return torch.linalg.norm(weighted_gt - weighted_avg, ord=2)

    def soft_histogram(self, data, weights, bin_centers, temperature=0.1):
        """
        data: (N, F) - standardized input data
        weights: (N,) - soft selection weights for each data point
        bin_centers: list of tensors of shape (K_f,) for each of F features
        temperature: float - sharpness of soft assignment (lower = sharper)
        
        Returns:
            hist: list of tensors, each (K_f,) representing soft histograms per feature
        """
        N, F = data.shape
        hist = []

        for i, f in enumerate(range(F)):
            if i != 3:
                continue
            x = data[:, f].unsqueeze(1)                    # (N, 1)
            centers = bin_centers[f].unsqueeze(0)          # (1, K)
            dist = -((x - centers)**2) / temperature       # (N, K) soft assignment scores
            assign = torch.nn.functional.softmax(dist, dim=1)                # (N, K) soft bin assignments
            weighted_assign = assign * weights.unsqueeze(1) # (N, K)
            hist_f = weighted_assign.sum(dim=0)            # (K,)
            hist_f = hist_f / (hist_f.sum() + 1e-8)        # Normalize
            hist.append(hist_f)

        return hist

    def jsd_soft(self, p_hists, q_hists):
        """
        p_hists, q_hists: lists of (K_f,) tensors representing histograms for each feature

        Returns:
            Scalar JSD value averaged over all features
        """
        jsd = 0.0
        for p, q in zip(p_hists, q_hists):
            m = 0.5 * (p + q)
            p_log = torch.log(torch.clamp(p, min=1e-8))
            q_log = torch.log(torch.clamp(q, min=1e-8))
            m_log = torch.log(torch.clamp(m, min=1e-8))

            jsd_f = 0.5 * (torch.sum(p * (p_log - m_log)) + torch.sum(q * (q_log - m_log)))
            jsd += jsd_f

        return jsd / len(p_hists)
    
    def get_JSD_loss(self,printt=False):
        jsd_total = 0.0
        weights = self.generator.get_weights_regularizer(self.bias_dataset).T.squeeze()
        gt_weights = (self.gt_weights / (self.gt_weights.sum() + 1e-12)).to(self.device)
        KLP_history = {}

        for i in range(self.unscaled_biased.shape[1]):
            # One-hot encode the i-th feature
            P = torch.nn.functional.one_hot(self.unscaled_biased[:, i].long(), num_classes=self.embedding_dict[i][1]).float()  # (N, num_classes)
            Q = torch.nn.functional.one_hot(self.unscaled_ground_truth[:, i].long(), num_classes=self.embedding_dict[i][1]).float()  # (K, num_classes)

            # Normalize to get empirical probability distributions
            P_dist = (weights[:, None] * P).sum(dim=0)          # (num_classes,)
            #Q_dist = Q.mean(dim=0)        # (num_classes,)
            Q_dist = (gt_weights[:, None] * Q).sum(dim=0) 

            M_dist = 0.5 * (P_dist + Q_dist)

            # Compute KL divergences (add small eps to avoid log(0))
            eps = 1e-8
            kl_P = torch.sum(P_dist * torch.log((P_dist + eps) / (M_dist + eps)))
            kl_Q = torch.sum(Q_dist * torch.log((Q_dist + eps) / (M_dist + eps)))
            KLP_history[i] = [P_dist.cpu(),Q_dist.cpu()]
            #scale by the dimensionality of the feature
            jsd_total += (kl_P + kl_Q) * (1 / (np.sqrt(self.embedding_dict[i][2]) - 1))

        if printt:
            print("weighted: ", KLP_history[0][0])
            print("census: ", KLP_history[0][1])
        #jsd_avg = jsd_total / self.unscaled_biased.shape[1]
        return jsd_total

    def L_KLIEP(self, eps=1e-8):
        """
        Compute KLIEP loss for categorical data.
        
        Args:
            X: Tensor of shape (n,), categorical indices (0..C-1)
            Y: Tensor of shape (m,), categorical indices (0..C-1)
            eps: Small value to avoid log(0)
        Returns:
            Scalar KLIEP loss (to minimize)
        """
        w = self.generator.get_weights_regularizer(self.bias_dataset).T.squeeze()

        X = self.unscaled_biased
        Y_whole = self.unscaled_ground_truth
        K = self.KLIEP_downsample
        Y_probs = self.gt_weights / self.gt_weights.sum()
        Y_probs = Y_probs.to(self.device)
        if K == -1:
            Y = Y_whole
        else:
            # Sample K rows from Y_whole according to Y_probs
            indices = torch.multinomial(Y_probs, num_samples=K, replacement=True, generator=self.rngs['torch'])
            Y = Y_whole[indices]

        # Compare each Y_j row with all X_i rows
        # indicator[i,j] = 1 if X[i] == Y[j] across all variables
        # Shape: (m, n)
         
        if False:
            #exact matching
            indicator = (Y.unsqueeze(1) == X.unsqueeze(0)).all(dim=2).float()  # shape (m, n)
            p_hat = torch.matmul(indicator, w)  # shape (m,)

            # Fraction of Y rows that have at least one matching X row
            #coverage_Y = (indicator.sum(dim=1) > 0).float().mean()
            # Fraction of X rows that have at least one matching Y row
            #coverage_X = (indicator.sum(dim=0) > 0).float().mean()
        else:
            #partial matching
            similarity = (Y.unsqueeze(1) == X.unsqueeze(0)).float().mean(dim=2)  # fraction of variables that match
            # Weighted sum over X for each Y_j
            p_hat = torch.matmul(similarity, w)  # shape (m,)

        # Avoid log(0)
        p_hat = p_hat + eps

        if K == -1:
            loss = -(Y_probs * torch.log(p_hat)).sum()
        else:
            # KLIEP loss
            loss = - torch.mean(torch.log(p_hat))

        return loss

    def get_JSD_loss_depricated(self):
        weights = self.generator.get_weights_regularizer(self.bias_dataset).T.squeeze()
        p_hist = self.soft_histogram(self.bias_dataset, weights, self.bin_centers)
        q_hist = self.soft_histogram(self.ground_truth_dataset, 
                                     torch.ones(self.ground_truth_dataset.size(0), device=self.ground_truth_dataset.device)/self.ground_truth_dataset.size(0), 
                                     self.bin_centers)

        loss_jsd = self.jsd_soft(p_hist, q_hist)
        return loss_jsd

    def train_generator(self):
        #bias_data, _ = self.generator(self.bias_dataset,self.temperature)
        selected_data, _ = self.generator.forward(self.bias_dataset)
        bias_label = self.discriminator(selected_data.to(self.device))
        data_loss = self.loss_function(bias_label, torch.ones_like(bias_label).to(self.device))
        
        #get regularizer loss here for the sum of the squared weights
        regularizer_loss = self.get_reg_loss()
        generator_loss = data_loss + self.lambda_weights * regularizer_loss

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()


        #warmup code
        with self.generator_warmup_scheduler.dampening():
            self.generator_lr_scheduler.step()

        return generator_loss.item(), regularizer_loss.item()

    def train_discriminator(self):

        ground_truth_data = self.generate_data()
        ground_truth_label = self.discriminator(ground_truth_data)
        bias_data, _ = self.generator(self.bias_dataset)
        bias_label = self.discriminator(bias_data.to(self.device))

        label_pred = torch.cat((ground_truth_label, bias_label), dim=0).to(self.device)
        label_true = torch.cat((torch.ones_like(ground_truth_label), torch.zeros_like(bias_label)), dim=0).to(self.device)
        
        with torch.no_grad():
            truth_loss = self.loss_function(ground_truth_label.detach(), torch.ones_like(ground_truth_label))
            bias_loss = self.loss_function(bias_label.detach(), torch.zeros_like(bias_label))

        indices = torch.randperm(label_pred.size(0))
        label_pred = label_pred[indices]
        label_true = label_true[indices]
        discriminator_loss = self.loss_function(label_pred, label_true)

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()

        #warmup code
        with self.discriminator_warmup_scheduler.dampening():
            self.discriminator_lr_scheduler.step()

        return discriminator_loss.item(), truth_loss.item(), bias_loss.item()
    
    def onestep_train(self, generator_training_factor=1, discriminator_training_factor=1):

        generator_losses = []
        discriminator_losses = []
        regularizer_losses = []
        truth_losses = []
        bias_losses = []
        for _ in range(generator_training_factor):
            generator_loss, regularizer_loss = self.train_generator()
            generator_losses.append(generator_loss)
            regularizer_losses.append(regularizer_loss)

        for _ in range(discriminator_training_factor):
            discriminator_loss, truth_loss, bias_loss = self.train_discriminator()
            discriminator_losses.append(discriminator_loss)
            truth_losses.append(truth_loss)
            bias_losses.append(bias_loss)

        return generator_losses, discriminator_losses, truth_losses, bias_losses, regularizer_losses

    def measure_correctness(self):
        l2_diff = None
        demographic_diff = 0
        if self.data_type == "real":
            weights = self.generator.get_weights(self.bias_dataset)
            predicted_target = (weights @ self.biased_labels.cpu().numpy()).item()
            l2_diff = np.abs(self.ground_truth - predicted_target)

            #get demographic difference here
            #self.ground_truth_demographics
            recovered_demographics = (weights @ self.bias_dataset.cpu().numpy()).flatten()
            demographic_diff = np.linalg.norm(recovered_demographics - self.ground_truth_demographics)
        elif self.data_type == "importance":
            l2_diff = np.linalg.norm(self.generator.get_weights(self.bias_dataset)-self.ground_truth)
            demographic_diff = 0
        else:
            raise NotImplementedError
        return l2_diff, demographic_diff
    
    def measure_HH_demopgrahics(self, printt):
        self.generator.set_eval()
        #get mean of census for each variable
        weighted_gt = np.average(self.gt_cpu, axis=0)
        #get mean of prediction with weights
        weights = self.generator.get_weights(self.bias_dataset)
        weighted_avg = np.average(self.bias_cpu, axis=0, weights=weights.flatten())
        #take l2 diff and return
        return np.linalg.norm(weighted_gt - weighted_avg, ord=2)
    
    def measure_entropy(self):
        self.generator.set_train()
        with torch.no_grad():
            probs = self.generator.get_weights(self.bias_dataset).flatten()
            max_entropy = np.log(float(len(probs)))
            entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy/max_entropy

    def measure_JS_divergence_OLD(self, printt):
        categories = np.union1d(np.unique(self.unscaled_ground_truth[:,-1]), np.unique(self.unscaled_biased[:,-1]))
        # Count weighted frequencies for A
        weights = self.generator.get_weights(self.bias_dataset).flatten()
        p = np.array([weights[self.unscaled_biased[:,-1].flatten() == cat].sum() for cat in categories])
        p = p / p.sum()  # normalize to get distribution
        # Count weighted frequencies for B
        w_B = np.array([1/(self.unscaled_ground_truth[:,-1].shape[0]) for _ in range(self.unscaled_ground_truth.shape[0])])
        q = np.array([w_B[self.unscaled_ground_truth[:,-1].flatten() == cat].sum() for cat in categories])
        q = q / q.sum()
        js_dist = jensenshannon(p, q)
        return js_dist

    def measure_intersection_recovery(self):
        col_indexes = self.col_indexes
        weights = self.generator.get_weights_regularizer(self.bias_dataset).T.squeeze().cpu()
        df = pd.DataFrame({
        'col_i': self.unscaled_biased[:, col_indexes[0]],
        'col_j': self.unscaled_biased[:, col_indexes[1]],
        'weight': weights})
        # Group by (col_i, col_j) and sum the weights
        joint = df.groupby(['col_i', 'col_j'])['weight'].sum()
        # Normalize to get a probability distribution
        joint_prob = joint / joint.sum()
        joint_prob = joint_prob.values
        #joint_prob = joint_prob.values.reshape((self.var_counts[0], self.var_counts[1]))
        
        if self.original_distribution is not None:
            jsd = jensenshannon(joint_prob, self.original_distribution, base=2) ** 2
            return jsd
        else:
            print(joint_prob)
            return 0

    def measure_vaccine(self):
        self.generator.set_eval()
        weights = self.generator.get_weights(self.bias_dataset)
        predicted_target = (weights @ self.biased_labels)
        return predicted_target, weights

    def measure_vaccine_batch(self):
        self.generator.set_eval()
        _, matrix, _ = self.generator.forward(self.bias_dataset)
        labels = matrix.cpu().numpy() @ self.biased_labels
        return np.mean(labels)

    def train(self,
              batchs_in_epoch,
              epochs,
              temperature_start,
              temperature_end,
              gen_training_factor,
              disc_training_factor,
              writer,):
        prob_diffs = []
        generator_losses, discriminator_losses = [], []
        test_probs = []
        test_prob_diffs = []
        for epoch in tqdm(range(epochs)):
            #TEMPERATURE = temperature_end + (temperature_start - temperature_end) * np.exp(-epoch/epochs)

            generator_losses_1ep, discriminator_loss_1ep, truth_losses, bias_losses, regularizer_losses = self.onestep_train(generator_training_factor=gen_training_factor, discriminator_training_factor=disc_training_factor)
            
            pred = self.measure_vaccine()
            #test_probs.append(l2_diff)
            writer.add_scalar('GenLoss', sum(generator_losses_1ep)/len(generator_losses_1ep), epoch)
            writer.add_scalar('Disc loss', sum(discriminator_loss_1ep)/len(discriminator_loss_1ep), epoch)
            writer.add_scalar('Vaccine prediction', pred, epoch)
            writer.add_scalar('regularizer losses', sum(regularizer_losses)/len(regularizer_losses),epoch)

        weights = self.generator.get_weights(self.bias_dataset)
        return weights, self.biased_labels, prob_diffs, test_probs, test_prob_diffs, generator_losses, discriminator_losses

    def save(self, epoch):
        if self.save_dict['SAVE_CRITIC']:
            torch.save(
                {
                    "config": self.critic_config,
                    "state_dict": {k: v.cpu() for k, v in self.discriminator.state_dict().items()}
                },
                "./"+self.save_dir+"/critic_checkpoint_"+str(epoch)+".pt"
            )

        if self.save_dict['SAVE_GENERATOR']: 
            torch.save(
                {
                    "config": self.generator_config,
                    "state_dict": {k: v.cpu() for k, v in self.generator.state_dict().items()}
                },
                "./"+self.save_dir+"/generator_checkpoint_"+str(epoch)+".pt"
            )
    
        if self.save_dict['SAVE_WEIGHTS']:
            weights = self.generator.get_weights(self.bias_dataset)
            #self.compute_input_gradient_generator()
            #np.savez("./"+self.save_dir+"/grad_history_"+str(trial_id)+"_"+str(synthetic_col_names)+".npz", 
            #        w=np.array(self.input_grad_history), wc=np.array(self.critic_input_grad_history))
            np.savez("./"+self.save_dir+"/weights_"+str(epoch)+".npz", 
                    w=np.array(weights))
        
        with open('./'+self.save_dir+'/entropy_history.pikl','wb') as file:
            pickle.dump(self.entropy_history, file)
                    
class WGAN_GP(GAN):

    def notebook_train(self,
                       num_epochs = 200,
                       gen_training_factor = 1,
                       disc_training_factor = 10):
        #purely for debugging purposes in jupyter notebook. not functional alone
        saved_demos = []
        for epoch in tqdm(range(num_epochs)):
            generator_losses_1ep, discriminator_loss_1ep, gps, gt_scores, bias_scores, unique_counts = self.onestep_train(generator_training_factor=gen_training_factor, discriminator_training_factor=disc_training_factor)
            out = self.measure_HH_demopgrahics()
            saved_demos.append(out)
        return saved_demos

    def autotune_train(self,
                        epochs,
                        gen_training_factor,
                        disc_training_factor,
                        writer,):
        '''
        Autotuner needs output:
            jsd_history : list[float]
            The jsd metric recorded at each step/epoch during training.
            Used by the tuner to assess convergence.

        kleip_history: list[float]
            The KLIEP metric recorded at each step/epoch during training.
            Used by the tuner to assess convergence.
        '''

        jsd_history = []
        kliep_history = []

        for epoch in tqdm(range(epochs)):              
            if epoch % 1 == 0:
                with torch.no_grad():
                    #obtain diagnostic information
                    self.generator.set_eval()
                    tvd_loss = self.L_KLIEP() #self.get_JSD_loss()
                    jsd_loss = self.get_JSD_loss(printt=False)
                    pred, _ = self.measure_vaccine()
                    entropy = self.measure_entropy()
                    reg_loss = self.get_reg_loss()

                    #write to tensorbaord
                    writer.add_scalar('tvd', tvd_loss.item(), epoch)
                    writer.add_scalar('jsd', jsd_loss.item(), epoch)
                    for pi in range(pred.shape[1]):
                        writer.add_scalar(str(self.label_names[pi]) + ' prediction', pred[0,pi], epoch)
                    writer.add_scalar('Gen Entropy', entropy, epoch)
                    writer.add_scalar('weights^2 loss', self.lambda_weights*reg_loss, epoch)

                    #append to return lists
                    kliep_history.append(tvd_loss.item())
                    jsd_history.append(jsd_loss.item())

            generator_losses_1ep, generator_spread_reg_1ep, discriminator_loss_1ep, tf_scores_1ep, gps, gt_scores, bias_scores, all_gp_grads, all_gen_grads, weights = self.onestep_train(generator_training_factor=gen_training_factor, 
                                                                                                                          discriminator_training_factor=disc_training_factor,
                                                                                                                          epoch=epoch,
                                                                                                                          total_epoch = epochs)



            writer.add_scalar('GenLoss', sum(generator_losses_1ep)/(len(generator_losses_1ep)+1e-6), epoch)
            writer.add_scalar('Spread Regularizer', sum(generator_spread_reg_1ep)/(len(generator_spread_reg_1ep)+1e-6), epoch)

            if len( tf_scores_1ep) > 0:
                writer.add_scalar('Truth - Fake scores', tf_scores_1ep[-1], epoch)

                writer.add_scalar('Disc Loss', discriminator_loss_1ep[-1],epoch)
                writer.add_scalar('GT score', sum(gt_scores)/(len(gt_scores)+1e-6),epoch)
                writer.add_scalar('Gradient Penalty', gps[-1], epoch)
                
                writer.add_scalar('Avg Gen Grad', sum(all_gen_grads)/(len(all_gen_grads)+1e-6),epoch)
                writer.add_scalar('All gp grads', all_gp_grads[-1],epoch)
            writer.add_scalar('Bias score', sum(bias_scores)/(len(bias_scores)+1e-6),epoch)

        return jsd_history, kliep_history

    
    def train(self,
              epochs,
              gen_training_factor,
              disc_training_factor,
              writer,
              trial_id,
              synthetic,
              synthetic_col_names = ""):
        
        if False: #lr scheduler code
            #lr scheduler for generator:
            self.generator.min_lr = 1e-7
            self.generator.reach_min_epoch = 100
            gamma = np.exp(np.log(self.generator.min_lr / self.generator_starting_lr) / self.generator.reach_min_epoch)
            self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.generator_optimizer, gamma=gamma)

            #lr scheduler for discminiator
            self.discriminator.min_lr = 1e-9
            reach_min_epoch = epochs - self.start_decay #(5 * epochs) // 10 #let it traing normally for first 50 epochs
            gamma = np.exp(np.log(self.discriminator.min_lr / self.discriminator_starting_lr) / reach_min_epoch)
            self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=gamma)
        
        prob_diffs = []
        generator_losses, discriminator_losses = [], []
        test_probs = []
        test_prob_diffs = []
        temp_disc_training_factor = disc_training_factor
        temp_gen_training_factor = gen_training_factor
        self.entropy_history = [] #reset entropy history
        pred_history = []
        weight_history = []
        jsd_history = []
        for epoch in tqdm(range(epochs)):
            if epoch % 1 == 0:
                with torch.no_grad():
                    self.generator.set_eval()
                    #out = self.measure_intersection_recovery()
                    #writer.add_scalar('tvd', out, epoch)

                    tvd_loss = self.L_KLIEP() #self.get_JSD_loss()
                    writer.add_scalar('tvd', tvd_loss.item(), epoch)
                    
                    if epoch % 100 == 0 or epoch+1 == epochs:
                        jsd_loss = self.get_JSD_loss(printt=False)
                    else:
                        jsd_loss = self.get_JSD_loss(printt=False)
                    jsd_history.append(jsd_loss.item())
                    writer.add_scalar('jsd', jsd_loss.item(), epoch)

                    #probs = self.generator.get_weights(self.bias_dataset).flatten()
                    #print(sum(probs[self.synthetic_label_1]),sum(probs[self.synthetic_label_2]))
                    #pred = self.measure_vaccine_batch()
                    #writer.add_scalar('Vaccine prediction', pred, epoch)
                    if not synthetic:
                        pred, _ = self.measure_vaccine()
                        pred_history.append(pred.item())
                        #weight_history.append(weights.flatten())
                        if len(pred.shape) == 1:
                            pred = np.expand_dims(pred,0)
                        for pi in range(pred.shape[1]):
                            writer.add_scalar(str(self.label_names[pi]) + ' prediction', pred[0,pi], epoch)
                        entropy = self.measure_entropy()
                        writer.add_scalar('Gen Entropy', entropy, epoch)
                        self.entropy_history.append(entropy)
                        reg_loss = self.get_reg_loss()
                        
                        writer.add_scalar('weights^2 loss', self.lambda_weights*reg_loss, epoch)
                        #sparse_loss = self.first_layer_sparse_loss()
                        #writer.add_scalar('first  layer sparse loss', sparse_loss.item(), epoch)

                        
                if False and self.save_weights:
                    #torch.save(self.generator, "./saves/generator"+str(epoch)+".pth")
                    weights_np = np.array(weight_history)
                    np.savez("./"+self.save_dir+"/weight_history_"+str(trial_id)+".npz", w=np.array(weight_history))

            generator_losses_1ep, generator_spread_reg_1ep, discriminator_loss_1ep, tf_scores_1ep, gps, gt_scores, bias_scores, all_gp_grads, all_gen_grads, weights = self.onestep_train(generator_training_factor=temp_gen_training_factor, 
                                                                                                                          discriminator_training_factor=temp_disc_training_factor,
                                                                                                                          epoch=epoch,
                                                                                                                          total_epoch = epochs)
            #self.compute_input_gradient_generator()
            #self.compute_input_gradient_disc()
            #exit(1)

            #if epoch >= 2 and not synthetic:
            #    writer.add_scalar('weight l2 hist', np.linalg.norm(weight_history[-1]-weight_history[-2],ord=2),epoch)
            writer.add_scalar('GenLoss', sum(generator_losses_1ep)/(len(generator_losses_1ep)+1e-6), epoch)
            writer.add_scalar('Spread Regularizer', sum(generator_spread_reg_1ep)/(len(generator_spread_reg_1ep)+1e-6), epoch)
            #writer.add_scalar('Truth - Fake scores', sum(discriminator_loss_1ep)/(len(discriminator_loss_1ep)+1e-6), epoch)
            #writer.add_scalar('Truth - Fake scores', discriminator_loss_1ep[-1], epoch)
            #for dl in range(len(tf_scores_1ep)):
            #    rdl = epoch*len(tf_scores_1ep)+dl
            if len( tf_scores_1ep) > 0:
                writer.add_scalar('Truth - Fake scores', tf_scores_1ep[-1], epoch)

                #for dl in range(len(discriminator_loss_1ep)):
                #    rdl = epoch*len(discriminator_loss_1ep)+dl
                
                writer.add_scalar('Disc Loss', discriminator_loss_1ep[-1],epoch)
                writer.add_scalar('GT score', sum(gt_scores)/(len(gt_scores)+1e-6),epoch)
                writer.add_scalar('Gradient Penalty', gps[-1], epoch)
                
                writer.add_scalar('Avg Gen Grad', sum(all_gen_grads)/(len(all_gen_grads)+1e-6),epoch)
                writer.add_scalar('All gp grads', all_gp_grads[-1],epoch)
            writer.add_scalar('Bias score', sum(bias_scores)/(len(bias_scores)+1e-6),epoch)
            
            #writer.add_scalar('Gen Learning rate', self.generator_optimizer.param_groups[0]['lr'],epoch)
            #writer.add_scalar('Disc Learning rate', self.discriminator_optimizer.param_groups[0]['lr'],epoch)
            #writer.add_scalar('Unique Counts', sum(unique_counts)/len(unique_counts), epoch)
            if self.save_dict['SAVE_EVERY']:
                if (self.save_dict['SAVE_EVERY'] != -1 
                    and epoch % self.save_dict['SAVE_EVERY'] == 0  
                    and epoch >= 300):
                    self.save(epoch=epoch)

        if any(self.save_dict.values()):
            with open("./"+self.save_dir+"/embedding_dict_"+str(trial_id)+"_"+str(synthetic_col_names)+".pikl",'wb') as file:
                pickle.dump(self.embedding_dict, file)

        if self.save_dict['SAVE_EVERY']:
            if self.save_dict['SAVE_EVERY'] == -1:
                self.save(epoch=epochs)
                    
        if self.save_dict['SAVE_IG']:
            self.compute_input_gradient_generator()
            np.savez("./"+self.save_dir+"/grad_history_"+str(trial_id)+".npz", 
            w=np.array(self.input_grad_history), wc=np.array(self.critic_input_grad_history))

        if self.save_dict['SAVE_DATASET']:
            np.savez("./"+self.save_dir+"/data_"+str(trial_id)+".npz", x=self.unscaled_biased.cpu().numpy(), 
                        y=self.biased_labels)
            np.savez("./"+self.save_dir+"/one_hot_data_"+str(trial_id)+".npz", x=self.bias_dataset.cpu().numpy(), 
                        y=self.biased_labels)
        
        

            
        return weights, self.biased_labels, prob_diffs, test_probs, test_prob_diffs, \
            generator_losses, discriminator_losses, jsd_history, pred_history

    def onestep_train(self, generator_training_factor=1, 
                      discriminator_training_factor=1,
                      epoch=0,
                      total_epoch=100):
        generator_losses = []
        spread_regularizer = []
        discriminator_losses = []
        gps = []
        gt_scores = []
        bias_scores = []
        all_gp_grads = []
        all_gen_grads = []
        tf_scores = []
        weight_history = []
        
        for _ in range(generator_training_factor):
            generator_loss, spread_reg, avg_grad, weights = self.train_generator(epoch)
            generator_losses.append(generator_loss)
            spread_regularizer.append(spread_reg)
            all_gen_grads.append(avg_grad)
            weight_history.append(weights.detach().cpu().numpy())
            if self.gen_history_length > 0:
                if len(self.weights_history) > self.gen_history_length:
                    self.weights_history.pop(0)
                self.weights_history.append(weights)
        
        #self.reset_discriminator()

        for _ in range(discriminator_training_factor):
            discriminator_loss, t_f_score, gp, gt_score, bias_score, gp_grads = self.train_discriminator(epoch)
            discriminator_losses.append(discriminator_loss)
            gps.append(gp)
            tf_scores.append(t_f_score)
            gt_scores.append(gt_score)
            bias_scores.append(bias_score)
            all_gp_grads.append(gp_grads)

        return generator_losses, spread_regularizer, discriminator_losses, tf_scores, gps, gt_scores, bias_scores, all_gp_grads, all_gen_grads, weight_history

    def train_generator_old(self, temperature_update = None):
        self.discriminator.set_eval()
        self.generator.set_train()

        #bias_data, _ = self.generator(self.bias_dataset,self.temperature)
        sds, _, _ = self.generator.forward(self.bias_dataset)
        #sds = torch.reshape(sds,(sds.shape[0],sds.shape[1]*sds.shape[2]))
        bias_scores = self.discriminator(sds.to(self.device))
        bias_loss = -torch.mean(bias_scores)
        regularizer_loss = self.get_reg_loss()
        demo_loss = 0
        if self.lambda_demo > 0:
            demo_loss = self.get_demo_loss()
        
        generator_loss = bias_loss + self.lambda_weights * regularizer_loss + self.lambda_demo * demo_loss
        #print(generator_loss.item(), bias_loss.item()) #(self.lambda_weights * regularizer_loss).item(), (self.lambda_demo * demo_loss).item())

        self.generator_optimizer.zero_grad()
        generator_loss.backward()

        total_grad = 0.0
        param_count = 0

        for param in self.generator.parameters():
            if param.grad is not None:
                total_grad += param.grad.abs().mean().item()
                param_count += 1

        avg_grad = total_grad / param_count if param_count > 0 else 0.0
        self.generator_optimizer.step()
        return generator_loss.item(), regularizer_loss.item(), avg_grad

    def first_layer_sparse_loss(self):
        pen = 0.0
        current_idx = 0
        for ind,info in self.embedding_dict.items():
            pen += info[2]**0.5 * self.generator.phi[0].weight[:,current_idx:current_idx+info[2]].pow(2).sum().sqrt()
            current_idx += info[2]
        return pen
    
    def train_generator(self, epoch, temperature_update = None):
        self.discriminator.set_train()
        self.generator.set_train()

        #bias_data, _ = self.generator(self.bias_dataset,self.temperature)
        sds, _, logits0 = self.generator.forward(self.bias_dataset)
        #sds = torch.reshape(sds,(sds.shape[0],sds.shape[1]*sds.shape[2]))
        bias_scores = self.discriminator.forward(sds.to(self.device))
        bias_loss = -torch.mean(bias_scores)
        
        weights = torch.nn.functional.softmax(logits0.flatten(),dim=0)
        spread_loss = 0
        demo_loss = 0
        demo_loss2 = 0
        first_l1 = 0
        if self.lambda_weights > 0:
            spread_loss = self.get_reg_loss() #self.get_medium_entropy_loss()
        if self.lambda_demo > 0:
            demo_loss = self.L_KLIEP() #self.get_JSD_loss()
        if self.lambda_jsd > 0:
            demo_loss2 = self.get_JSD_loss()
        #if self.lambda_first_layer > 0:
        #    first_l1 = self.first_layer_sparse_loss()
        if True:
            generator_loss = bias_loss \
                            + self.lambda_weights * spread_loss \
                            + self.lambda_demo * demo_loss \
                            + self.lambda_jsd * demo_loss2
        else:
            generator_loss = self.lambda_jsd * demo_loss2 #+ self.lambda_demo * demo_loss
        #print(self.lambda_first_layer * first_l1)
        #print("values below T=0.1", (self.generator.phi[0].weight < 0.1).sum().item())
        #print(bias_loss.item(), self.lambda_demo * demo_loss)
        #exit(1)
        #print(generator_loss)

        if False:
            with torch.no_grad():
                p0 = {}
                for n, p in self.generator.named_parameters():
                    print(n)
                    if p.requires_grad:
                        p0[n] = p.detach().clone()


        self.generator_optimizer.zero_grad()
        generator_loss.backward()

        if not isinstance(spread_loss, int):
            spread_loss = spread_loss.item()

        avg_grad = -1 #
        #avg_grad = np.mean([get_avg_grad_per_layer(self.generator.rho), get_avg_grad_per_layer(self.generator.phi)])
        self.generator_optimizer.step()

        # after step
        if False:
            with torch.no_grad():
                _, _, logits1 = self.generator(self.bias_dataset)
                rint(weights)
                weights1 = torch.nn.functional.softmax(logits1.flatten(),dim=0)
                print(weights1)
                delta = (logits1 - logits0).abs()
                print("mean |Δlogits|:", delta.mean().item())
                print("max  |Δlogits|:", delta.max().item())
                print("mean |logits|:", logits0.abs().mean().item())

                print("")
                delta2 = (weights1 - weights).abs()
                print("mean |Δweights|:", delta2.mean().item())
                print("max  |Δweights|:", delta2.max().item())
                print("mean |Δweights|:", weights.abs().mean().item())
            exit(1)

            with torch.no_grad():
                deltas = []
                for n, p in self.generator.named_parameters():
                    if n in p0:
                        deltas.append((n, (p - p0[n]).abs().mean().item()))
                deltas.sort(key=lambda t: t[1], reverse=True)
                print("Top parameter mean |Δ|:")
                for n, d in deltas[:10]:
                    print(f"  {n:60s} {d:.3e}")
            exit(1)
        #print(self.generator.logit_scale.item())
        return generator_loss.item(), spread_loss, avg_grad, weights
    
    def train_discriminator(self, epoch, override_weights=None):
        self.discriminator.set_train()
        self.generator.set_train()
        sds = []
        bias_unique_counts = [0]
        sds, selected_indices, weights = self.generator.forward(self.bias_dataset)
        if True: #weights history
            for past_weight in self.weights_history:
                indices = torch.multinomial(past_weight, self.sample_size, replacement=False)
                sds_past = self.bias_dataset[indices].unsqueeze(0)
                sds = torch.concat((sds,sds_past))
        bias_scores = self.discriminator(sds.to(self.device))
        bo = None #sds.shape[0]
        ground_truth_data = self.generate_data(batch_override=bo)
        ground_truth_scores = self.discriminator(ground_truth_data)
        #print(torch.mean(bias_scores), torch.mean(ground_truth_scores))
        # Compute WGAN-GP loss
        gp = 0
        gp_grad = 0
        if self.lambda_gp > 0:
            gp, gp_grad = self.compute_penalty(ground_truth_data, sds, (bo is not None))
        discriminator_loss = torch.mean(bias_scores) - torch.mean(ground_truth_scores) + self.lambda_gp * gp
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward() #removed retain_graph = True
        #print(torch.mean(bias_scores).item(), torch.mean(ground_truth_scores).item(), gp)
        if not isinstance(gp, int):
            gp = gp.item()

        #clipping disc
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10)

        self.discriminator_optimizer.step()

        #warmup code
        #with self.discriminator_warmup_scheduler.dampening():
            #current_lr = self.discriminator_lr_scheduler.get_last_lr()[0]
            #if current_lr > self.discriminator.min_lr:
            #self.discriminator_lr_scheduler.step()

        return discriminator_loss.item(), (torch.mean(ground_truth_scores) - torch.mean(bias_scores)).item(), self.lambda_gp * gp, torch.mean(ground_truth_scores).item(), torch.mean(bias_scores).item(), gp_grad

    def compute_penalty_old(self, ground_truth_data, bias_data):

        batch_size, *rest = ground_truth_data.shape
        epsilon = torch.rand(batch_size, 1, 1, device=self.device, generator=self.rngs['torch_cuda']).expand_as(ground_truth_data)
        
        # Create interpolated samples
        interpolated = (epsilon * ground_truth_data + (1 - epsilon) * bias_data).requires_grad_(True)
        # Get critic scores
        interpolated_scores = self.discriminator(interpolated.to(self.device))
        
        # Compute gradients
        gradients = autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
    
    def compute_penalty(self, ground_truth_data, bias_data, traditional=True):
        batch_size, *rest = bias_data.shape
        if True: #traditional:
            epsilon = torch.rand(batch_size, 1, 1, device=self.device, generator=self.rngs['torch_cuda']).expand_as(bias_data)
            interpolated = (epsilon * ground_truth_data + (1 - epsilon) * bias_data).requires_grad_(True)
        else: #just around bias data
            noise_std = 0.1  # you can tune this
            interpolated = (bias_data + noise_std * torch.randn_like(bias_data)).requires_grad_(True)
        interpolated_scores = self.discriminator(interpolated)
        interpolated_scores = interpolated_scores.view(-1)
        # Compute gradients
        grad_outputs = torch.ones_like(interpolated_scores, device=self.device)

        gradients = autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        
        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty, gradients.detach().norm(2, dim=1).mean().item()
    
    def get_temperature(self, epoch, EPOCH, init_temp=1.0, mid_temp=0.6, final_temp=0.3):
        """
        Returns a temperature value for the current training epoch following a 3-phase schedule:
        - Early: explore with high temp
        - Middle: moderate focus
        - Late: sharpen selections
        
        Args:
            epoch (int): Current training epoch (0-based).
            EPOCH (int): Total number of training epochs.
            init_temp (float): Initial high temperature (default=1.0).
            mid_temp (float): Mid-range temperature (default=0.6).
            final_temp (float): Final low temperature (default=0.3).
        
        Returns:
            float: Temperature value for the current epoch.
        """
        phase1_end = int(0.3 * EPOCH)
        phase2_end = int(0.8 * EPOCH)

        if epoch <= phase1_end:
            # Linear decay from init_temp to mid_temp
            alpha = epoch / phase1_end
            temp = (1 - alpha) * init_temp + alpha * mid_temp
        elif epoch <= phase2_end:
            # Linear decay from mid_temp to final_temp
            alpha = (epoch - phase1_end) / (phase2_end - phase1_end)
            temp = (1 - alpha) * mid_temp + alpha * final_temp
        else:
            # Constant temperature in the final phase
            temp = final_temp

        return temp

class WGAN_GP_fict(WGAN_GP):
    def __init__(self, history_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_length = history_length
        t_copy = copy.deepcopy(self.generator)
        self.gen_history = [t_copy]
        #self.gen_history[0].eval()

    def train_generator(self):
        self.discriminator.set_eval()
        self.generator.set_train()
        #bias_data, _ = self.generator(self.bias_dataset,self.temperature)
        sds, _, _ = self.generator.forward(self.bias_dataset)
        #sds = torch.reshape(sds,(sds.shape[0],sds.shape[1]*sds.shape[2]))
        bias_scores = self.discriminator(sds.to(self.device))
        generator_loss = -torch.mean(bias_scores)
        regularizer_loss = self.get_medium_entropy_loss()
        demo_loss = self.get_JSD_loss()
        generator_loss = generator_loss \
                        + self.lambda_weights * regularizer_loss \
                        + self.lambda_demo * demo_loss

        self.generator_optimizer.zero_grad()
        generator_loss.backward()

        avg_grad = 0
        if False: #old grad checking debugging code
            total_grad = 0.0
            param_count = 0

            for param in self.generator.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().mean().item()
                    param_count += 1

            avg_grad = total_grad / param_count if param_count > 0 else 0.0

        self.generator_optimizer.step()

        self.gen_history.append(copy.deepcopy(self.generator))
        if len(self.gen_history) > self.history_length:
            self.gen_history.pop(0)
            self.gen_history[-1].set_eval()
        #warmup code
        #with self.generator_warmup_scheduler.dampening():
        #    self.generator_lr_scheduler.step()

        return generator_loss.item(), regularizer_loss.item(), avg_grad
    
    def train_discriminator(self, epoch):
        self.discriminator.set_train()
        ground_truth_data = self.generate_data()
        ground_truth_scores = self.discriminator(ground_truth_data)
        sds = []
        batch_override = self.batch_size//(len(self.gen_history))
        for gen in self.gen_history[:-1]:
            if batch_override != 0:
                g_sds, _, _ = gen.forward(self.bias_dataset, batch_override=batch_override)
                sds.append(g_sds)
        remainder = self.batch_size - batch_override*len(self.gen_history[:-1])
        sds.append(self.gen_history[-1].forward(self.bias_dataset, batch_override=remainder)[0])
        sds = torch.concat(sds,dim=0)
        bias_scores = self.discriminator(sds.to(self.device))
        # Compute WGAN-GP loss
        gp = self.compute_penalty(ground_truth_data, sds)
        discriminator_loss = torch.mean(bias_scores) - torch.mean(ground_truth_scores) + self.lambda_gp * gp
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward() #removed retain_graph = True

        #clipping disc
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10)

        self.discriminator_optimizer.step()

        #warmup code
        #with self.discriminator_warmup_scheduler.dampening():
            #current_lr = self.discriminator_lr_scheduler.get_last_lr()[0]
            #if current_lr > self.discriminator.min_lr:
            #self.discriminator_lr_scheduler.step()

        return (torch.mean(ground_truth_scores) - torch.mean(bias_scores)).item(), self.lambda_gp * gp.item(), torch.mean(ground_truth_scores).item(), torch.mean(bias_scores).item(), 0


import torch
import torch.nn as nn
import numpy as np
import random
from Generator import onesGen, dataGen, weightsGen
from Discriminator import Discriminator
from util import dict2vector
from tqdm import tqdm
import pytorch_warmup as warmup

class GAN():
    def __init__(self,
                 dataset,
                 generator_type,
                gen_learning_rate,
                disc_learning_rate,
                truth_sample_size,
                gen_layers,
                bias_sample_size,
                temperature=0.1,
                warmup_length = 1000,
                device=torch.device('cuda:0'),
                ):

        '''
        subset_size,
                 batch_size,
                 temperature,
                 output_size,
                 device
        '''
        self.device=device
        self.temperature = temperature
        self.generator_type = generator_type
        if generator_type == 'onesGen':
            self.generator = onesGen(layers=gen_layers,
                                        sample_size=bias_sample_size,
                                        batch_size = 1,
                                        temperature=self.temperature,
                                        output_size = len(dataset.biased_dataset),
                                        device=self.device)
        elif generator_type == 'dataGen':
            self.generator = dataGen(num_features=dataset.biased_dataset.shape[1],
                                     layers=gen_layers,
                                     sample_size=bias_sample_size,
                                     temperature=temperature).to(self.device)
        elif generator_type == 'weightsGen':
            self.generator = weightsGen(dataset.biased_dataset.shape[0],
                                            bias_sample_size,).to(self.device)
        else:
            raise NotImplementedError
        self.discriminator = Discriminator(dataset.biased_dataset.shape[1]).to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                    lr=gen_learning_rate,
                                                    weight_decay=0)
        self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.generator_optimizer, gamma=1)
        self.generator_warmup_scheduler = warmup.LinearWarmup(self.generator_optimizer, warmup_length)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=disc_learning_rate,
                                                        weight_decay=0)
        self.ground_truth_dataset = dataset.ground_truth_dataset
        self.bias_dataset = dataset.biased_dataset
        self.ground_truth = dataset.ground_truth
        self.truth_sample_size = truth_sample_size    
        self.data_type = dataset.type

        self.biased_labels = dataset.biased_labels   

    def set_temperature(self, new_temp):
        self.temperature = new_temp

    def generate_data(self):
        index = np.random.choice(np.arange(len(self.ground_truth_dataset)),size=self.truth_sample_size)
        return self.ground_truth_dataset[index]

    def train_generator(self):
        #bias_data, _ = self.generator(self.bias_dataset,self.temperature)
        selected_data, _ = self.generator.forward(self.bias_dataset)
        bias_label = self.discriminator(selected_data.to(self.device))
        generator_loss = self.loss_function(bias_label, torch.ones_like(bias_label).to(self.device))
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        #warmup code
        with self.generator_warmup_scheduler.dampening():
            self.generator_lr_scheduler.step()

        return generator_loss.item()

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

        return discriminator_loss.item(), truth_loss.item(), bias_loss.item()
    
    def onestep_train(self, generator_training_factor=1, discriminator_training_factor=1):

        generator_losses = []
        discriminator_losses = []
        truth_losses = []
        bias_losses = []
        for _ in range(generator_training_factor):
            generator_loss = self.train_generator()
            generator_losses.append(generator_loss)

        for _ in range(discriminator_training_factor):
            discriminator_loss, truth_loss, bias_loss = self.train_discriminator()
            discriminator_losses.append(discriminator_loss)
            truth_losses.append(truth_loss)
            bias_losses.append(bias_loss)

        return generator_losses, discriminator_losses, truth_losses, bias_losses

    def measure_correctness(self):
        l2_diff = None
        if self.data_type == "real":
            test = self.generator.get_weights(self.bias_dataset)
            test2 = (test @ self.biased_labels.cpu().numpy()).item()
            l2_diff = np.abs(self.ground_truth - test2)
        elif self.data_type == "importance":
            l2_diff = np.linalg.norm(self.generator.get_weights(self.bias_dataset)-self.ground_truth)
        else:
            raise NotImplementedError
        return l2_diff
    
    def train(self,
              batchs_in_epoch,
              epochs,
              temperature_start,
              temperature_end,
              gen_training_factor,
              disc_training_factor,
              writer,):
        clusters, probs, prob_diffs = [], [], []
        generator_losses, discriminator_losses = [], []
        test_probs = []
        test_prob_diffs = []
        for epoch in tqdm(range(epochs)):
            #TEMPERATURE = temperature_end + (temperature_start - temperature_end) * np.exp(-epoch/epochs)
            generator_losses_1ep, discriminator_loss_1ep, truth_losses, bias_losses = self.onestep_train(generator_training_factor=gen_training_factor, discriminator_training_factor=disc_training_factor)
            
            l2_diff = self.measure_correctness()
            test_probs.append(l2_diff)
            writer.add_scalar('GenLoss', sum(generator_losses_1ep)/len(generator_losses_1ep), epoch)
            writer.add_scalar('Disc Truth loss', sum(truth_losses)/len(truth_losses), epoch)
            writer.add_scalar('Disc Bias loss', sum(bias_losses)/len(bias_losses), epoch)
            writer.add_scalar('L2 Diff Norm', l2_diff,epoch)

            if False and epoch % 100 == 0:
                print("")
                print(epoch)
                print(l2_diff)
                print(self.generator.get_weights(self.bias_dataset))
                print(self.ground_truth)
        return clusters, probs, prob_diffs, test_probs, test_prob_diffs, generator_losses, discriminator_losses

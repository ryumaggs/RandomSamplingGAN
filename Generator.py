import torch
import torch.nn as nn
import torch.nn.functional as F


class onesGen(torch.nn.Module):
    '''
    Lowest performer by far, no longer using 
    '''
    def __init__(self,
                 layers,
                 sample_size,
                 batch_size,
                 temperature,
                 output_size,
                 device):
        super().__init__()
        dropout = 0.1
        modules = []
        if layers is None:
            modules.append(nn.Linear(1,output_size))
        else:
            modules.append(nn.Linear(1,layers[0]))
            modules.append(nn.Dropout(dropout))
            modules.append(nn.LeakyReLU())
            for i,l in enumerate(layers):
                if i == 0:
                    continue
                modules.append(nn.Linear(layers[i-1],l))
                modules.append(nn.Dropout(dropout))
                modules.append(nn.LeakyReLU())
            modules.append(nn.Linear(layers[-1],output_size))
            modules.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*modules)
        self.device=device
        self.model.to(self.device)
        self.batch_size=batch_size
        self.sample_size=sample_size
        self.temperature = temperature

    def forward(self,tensor_dataset):
        '''
        dataset - is the data set cast as a torch tensor and moved to the appropriate device.

        samples indexes using gumbel softmax + params

        matrix multiples the one-hot-encoded indexes with data set
        to differentiably isolate SUBSET_SIZE data points
        '''
        
        batch = []
        for _ in range(self.batch_size):
            input = torch.tensor([[1] for _ in range(self.sample_size)], dtype=torch.float32).to(self.device)
            input = input.reshape(self.sample_size, 1)
            logits = self.model(input)
            indexes = F.gumbel_softmax(logits, tau=self.temperature, hard=False) #picks from the logits
            output = indexes @ tensor_dataset
            batch.append(output)
        return torch.stack(batch,axis=0).squeeze(), indexes.detach().cpu().numpy()
    
    def forward_debug(self,tensor_dataset):
        '''
        dataset - is the data set cast as a torch tensor and moved to the appropriate device.

        samples indexes using gumbel softmax + params

        matrix multiples the one-hot-encoded indexes with data set
        to differentiably isolate SUBSET_SIZE data points
        '''
        
        for _ in range(self.batch_size):
            input = torch.tensor([[1] for _ in range(self.subset_size)], dtype=torch.float32).to(self.device)
            input = input.reshape(self.subset_size, 1)
            logits = self.linear(input)
            indexes = F.gumbel_softmax(logits, tau=self.temperature, hard=False) #picks from the logits
            output = indexes @ tensor_dataset
            
        return output.flatten().unsqueeze(0)
    
    def get_weights(self, dummy_input):
        '''
        dummy_input exists to make code congruant with Generator() class
        '''
        with torch.no_grad():
            input = torch.tensor([[1]], dtype=torch.float32).to(self.device)
            logits = self.model(input)
            softmaxed = torch.nn.functional.softmax(logits,dim=1)
        return softmaxed.cpu().numpy()

class dataGen(nn.Module):
    def __init__(self,
                 num_features,
                 sample_size,
                 temperature):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sample_size = sample_size
        self.temperature=temperature

    def forward(self, dataset):
        logits = None
        logits = self.linear(dataset).T
        logits = logits.repeat((self.sample_size,1))
        matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=False) # give index
        output = torch.matmul(matrix, dataset)
        with torch.no_grad():
            max_indices = torch.argmax(matrix, dim=1)
        return output, max_indices

    def get_weights(self, dataset):
        with torch.no_grad():
            logit = self.linear(dataset).transpose(0,1)
            output = nn.Softmax(dim=1)(logit)
        return output.cpu().numpy()

class weightsGen(torch.nn.Module):
    def __init__(self,num_data_points,sample_size,):
        super().__init__()
        self.sample_size=sample_size
        self.weights = nn.Parameter(torch.rand((1, num_data_points), requires_grad=True))
    def forward(self, tensor_dataset):
        softmaxed_weights = torch.nn.functional.log_softmax(self.weights,dim=1)
        indexes = F.gumbel_softmax(softmaxed_weights.repeat(self.sample_size,1),
                                   tau=0.1,hard=False)
        return indexes @ tensor_dataset, None
    
    def get_weights(self, dummy_input):
        with torch.no_grad():
            return torch.nn.functional.softmax(self.weights,dim=1).detach().cpu().numpy()
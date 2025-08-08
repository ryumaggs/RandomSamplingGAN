import torch
import torch.nn as nn
import torch.nn.functional as F


class onesGen(torch.nn.Module):
    '''
    Lowest performer by far, no longer using 
    '''
    def __init__(self,
                 seed,
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
                 rng,
                 num_features,
                 layers,
                 sample_size,
                 dropout,
                 temperature):
        super().__init__()

        #self.linear = nn.Linear(num_features, 1)
        self.rng = rng
        modules = []
        if layers is None:
            modules.append(nn.Linear(num_features, 1))
        else:
            modules.append(nn.Linear(num_features,layers[0]))
    
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))
            for i,l in enumerate(layers):
                if i == 0:
                    continue
                modules.append(nn.Linear(layers[i-1],l))
                modules.append(nn.LeakyReLU())
                modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layers[-1],1))
            modules.append(nn.Identity())
        self.model = nn.Sequential(*modules)
        self.apply_he_init_to_sequential(self.model)
        self.sample_size = sample_size
        self.temperature=temperature

    def apply_he_init_to_sequential(self,model):
        for layer_id, layer in enumerate(model):
            #initialize all intermediary layers using relu non linearity
            if isinstance(layer, torch.nn.Linear):
                if isinstance(model[layer_id+1],torch.nn.LeakyReLU):  # Apply He initialization to Linear layers
                    torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='leaky_relu', generator=self.rng)  # or kaiming_uniform_
                elif isinstance(model[layer_id+1],torch.nn.Identity):
                    torch.nn.init.xavier_uniform_(layer.weight, generator=self.rng) 
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)  # Initialize biases to 0

    def forward(self, dataset):
        logits = self.model(dataset).T
        logits = logits.repeat((self.sample_size,1))
        matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=False) # give index
        output = torch.matmul(matrix, dataset)
        with torch.no_grad():
            max_indices = torch.argmax(matrix, dim=1)
        return output, max_indices, logits.detach().cpu()

    def get_weights(self, dataset):
        with torch.no_grad():
            logit = self.model(dataset).transpose(0,1)
            output = nn.Softmax(dim=1)(logit)
        return output.cpu().numpy()

    def get_weights_regularizer(self, dataset):
        logit = self.model(dataset).transpose(0,1)
        output = nn.Softmax(dim=1)(logit)
        return output
    
class weightsGen(torch.nn.Module):
    def __init__(self,seed,num_data_points,sample_size,):
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
        
class DeepSetNet(nn.Module):
    def __init__(self,
                 rngs, 
                 num_features,
                 layers,
                 sample_size,
                 dropout,
                 temperature,
                 batch_size=1,
                 hidden_dim=1024,):
        '''
        commented code for fast experimentation
        '''
        super().__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.temperature = temperature
        self.rngs = rngs
        phi_layers = []
        rho_layers = []
        if len(layers) == 0:
            phi_layers.append(nn.Linear(num_features,hidden_dim))
            phi_layers.append(nn.Identity())

            rho_layers.append(nn.Linear(hidden_dim, 1))
            rho_layers.append(nn.Identity())
        else:
            phi_layers.append(nn.Linear(num_features,layers[0]))
            #phi_layers.append(nn.LayerNorm(layers[0]))
            phi_layers.append(nn.LeakyReLU(0.2))
            phi_layers.append(nn.Dropout(dropout))

            rho_layers.append(nn.Linear(hidden_dim,layers[0]))
            #rho_layers.append(nn.LayerNorm(layers[0]))
            rho_layers.append(nn.LeakyReLU(0.2))
            rho_layers.append(nn.Dropout(dropout))
            for i,l in enumerate(layers):
                if i+1 >= len(layers): #at the end
                    phi_layers.append(nn.Linear(l,hidden_dim))
                    phi_layers.append(nn.Identity())

                    rho_layers.append(nn.Linear(l,1))
                    rho_layers.append(nn.Identity())
                else:
                    phi_layers.append(nn.Linear(l,layers[i+1]))
                    #phi_layers.append(nn.LayerNorm(layers[i+1]))
                    phi_layers.append(nn.LeakyReLU(0.2))
                    phi_layers.append(nn.Dropout(dropout))

                    rho_layers.append(nn.Linear(l,layers[i+1]))
                    #rho_layers.append(nn.LayerNorm(layers[i+1]))
                    rho_layers.append(nn.LeakyReLU(0.2))
                    rho_layers.append(nn.Dropout(dropout))
        self.phi = nn.Sequential(*phi_layers)
        self.rho = nn.Sequential(*rho_layers)

        self.apply_he_init_to_sequential(self.phi)
        self.apply_he_init_to_sequential(self.rho)

        self.phi.to(torch.device('cuda:0'))
        self.rho.to(torch.device('cuda:0'))

    def apply_he_init_to_sequential(self,model):
        for layer_id, layer in enumerate(model):
            #print(layer_id, layer)
            #initialize all intermediary layers using relu non linearity
            if isinstance(layer, torch.nn.Linear):
                if isinstance(model[layer_id+1],torch.nn.Identity):
                    torch.nn.init.xavier_uniform_(layer.weight, generator=self.rngs['torch']) 
                elif isinstance(model[layer_id+1],torch.nn.LeakyReLU):
                    torch.nn.init.kaiming_uniform_(layer.weight, generator=self.rngs['torch']) 
                elif isinstance(model[layer_id+2],torch.nn.Identity):
                    torch.nn.init.xavier_uniform_(layer.weight, generator=self.rngs['torch']) 
                elif isinstance(model[layer_id+2],torch.nn.LeakyReLU):  # Apply He initialization to Linear layers
                    torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='leaky_relu', generator=self.rngs['torch'])
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)  # Initialize biases to 0
            elif isinstance(layer, torch.nn.LayerNorm):
                if layer.elementwise_affine:
                    torch.nn.init.uniform_(layer.weight, a=0.0, b=1.0, generator=self.rngs['torch'])  # or any init you prefer
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, dataset, batch_override=None):  # x: [N, 8]
        bs = self.batch_size
        if batch_override:
            bs = batch_override
        x_phi = self.phi(dataset)                     # [N, hidden_dim]
        global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        x_rho = self.rho(x_phi + global_context)          # [N, 1]
        logits = x_rho.squeeze(1)
        #probs = F.softmax(x_rho.squeeze(-1), dim=0)       # [N]
        logits = logits.unsqueeze(0).repeat((self.sample_size*bs,1))
        #matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1) # give index
        # manual gumbel softmax
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=self.rngs['torch_cuda']).log()
        # Apply the Gumbel-Softmax transformation
        y = (logits + gumbels) / self.temperature
        matrix = y.softmax(dim=-1)
        output = torch.matmul(matrix, dataset)
        output = output.reshape(bs,self.sample_size,dataset.shape[1])
        return output, matrix.detach(), logits[0].detach().cpu()
    
    def get_weights(self, dataset):
        with torch.no_grad():
            x_phi = self.phi(dataset)                     # [N, hidden_dim]
            global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            x_rho = self.rho(x_phi + global_context).T          # [1, N]
            output = F.softmax(x_rho, dim=1)       # [N]
        return output.cpu().numpy()
    
    def get_weights_regularizer(self, dataset):
        x_phi = self.phi(dataset)                     # [N, hidden_dim]
        global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        x_rho = self.rho(x_phi + global_context).T          # [1, N]
        output = F.softmax(x_rho, dim=1)       # [N]
        return output
    
    def set_eval(self):
        self.phi.eval()
        self.rho.eval()
    
    def set_train(self):
        self.phi.train()
        self.rho.train()
    
    def update_temperature(self, new_temp):
        self.temperature = new_temp
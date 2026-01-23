import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
from ComplexNet import *
sys.path.append(os.path.dirname(__file__))

from util import embed_data, create_embedding_layers

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
                 embedding_dict,
                 batch_size=1,
                 hidden_dim=1024,):
        '''
        commented code for fast experimentation
        '''
        super().__init__()
        self.embedding_dict = embedding_dict
        self.embedding_layers = create_embedding_layers(self.embedding_dict,torch.device('cuda:0'))
        
        
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

    def forward(self, dataset, batch_override=None): 
        '''
        dataset is (1 x N x D) matrix, not [N x D]
        '''
        bs = self.batch_size
        #handle embeddings
        #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
        if batch_override:
            bs = batch_override
        x_phi = self.phi(dataset)                
        global_context = x_phi.mean(dim=1, keepdim=True) 
        x_rho = self.rho(x_phi + global_context)          # [N, 1]
        logits = x_rho.squeeze(2)
        #temp log layer for testing
        #logits = torch.log(logits)
        #probs = F.softmax(x_rho.squeeze(-1), dim=0)       # [N]
        logits_rep = logits.repeat((self.sample_size*bs,1))
        #matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1) # give index
        # manual gumbel softmax
        gumbels = -torch.empty_like(logits_rep, memory_format=torch.legacy_contiguous_format).exponential_(generator=self.rngs['torch_cuda']).log()
        # Apply the Gumbel-Softmax transformation
        y = (logits_rep + gumbels) / self.temperature
        
        matrix = y.softmax(dim=-1)
        output = torch.matmul(matrix, dataset)
        output = output.reshape(bs,self.sample_size,dataset.shape[2])
        return output, matrix.detach(), logits

    def get_weights(self, dataset):
        self.eval()
        with torch.no_grad():
            #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
            x_phi = self.phi(dataset)                     # [N, hidden_dim]
            global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            x_rho = self.rho(x_phi + global_context)#.squeeze().unsqueeze(0)            # [1, N]
            output = F.softmax(x_rho, dim=1)       # [N]
        return output.cpu().squeeze(0).numpy().T
    
    def get_logits(self, dataset):
        self.eval()
        with torch.no_grad():
            #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
            x_phi = self.phi(dataset)                    
            global_context = x_phi.mean(dim=0, keepdim=True)  
            x_rho = self.rho(x_phi + global_context)
            output = x_rho       # [N]
        return output.cpu().numpy()
    
    def get_weights_regularizer(self, dataset):
        self.train()
        #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
        x_phi = self.phi(dataset)                     # [N, hidden_dim]
        global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        x_rho = self.rho(x_phi + global_context).squeeze().unsqueeze(0)          # [1, N]
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

class DeepSetComplexNet(DeepSetNet):
    def __init__(self,
             rngs,
             num_features,
             layers,
             sample_size,
             dropout,
             temperature,
             embedding_dict,
             batch_size=1,
             hidden_dim=1024):
        super().__init__(
            rngs,
            num_features,
            layers,
            sample_size,
            dropout,
            temperature,
            embedding_dict,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )

        self.embedding_dict = embedding_dict
        self.embedding_layers = create_embedding_layers(self.embedding_dict, torch.device("cuda:0"))

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.temperature = temperature
        self.rngs = rngs

        # LayerNorm applied to the pooled set representation (phi -> rho interface)
        self.hidden_ln = nn.LayerNorm(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        def add_block(seq, in_dim, out_dim, add_norm_act_dropout: bool):
            """
            Appends a ComplexLayer(in_dim->out_dim) and optionally LN + LeakyReLU + Dropout.
            For the final layer, set add_norm_act_dropout=False.
            """
            seq.append(ComplexLayer(in_dim, out_dim))
            if add_norm_act_dropout:
                seq.append(nn.LayerNorm(out_dim))
                seq.append(nn.LeakyReLU(0.2))
                seq.append(nn.Dropout(dropout))
            else:
                seq.append(nn.Identity())

        # Decide the hidden widths for intermediate layers
        # Example: if layers=[256,128], then phi: num_features->256->128->hidden_dim
        widths = list(layers) if len(layers) > 0 else []

        # ---- Build PHI ----
        phi_layers = []
        if len(widths) == 0:
            # minimal phi: num_features -> hidden_dim
            add_block(phi_layers, num_features, hidden_dim, add_norm_act_dropout=False)
        else:
            # first: num_features -> widths[0] (with LN/act/dropout)
            add_block(phi_layers, num_features, widths[0], add_norm_act_dropout=True)
            # middle: widths[i] -> widths[i+1]
            for in_d, out_d in zip(widths[:-1], widths[1:]):
                add_block(phi_layers, in_d, out_d, add_norm_act_dropout=True)
            # final: widths[-1] -> hidden_dim (no LN/act/dropout, but Identity to keep pattern)
            add_block(phi_layers, widths[-1], hidden_dim, add_norm_act_dropout=False)

        self.phi = nn.Sequential(*phi_layers)

        # ---- Build RHO ----
        rho_layers = []
        if len(widths) == 0:
            # minimal rho: hidden_dim -> 1
            add_block(rho_layers, hidden_dim, 1, add_norm_act_dropout=False)
        else:
            # first: hidden_dim -> widths[0] (with LN/act/dropout)
            add_block(rho_layers, hidden_dim, widths[0], add_norm_act_dropout=True)
            # middle: widths[i] -> widths[i+1]
            for in_d, out_d in zip(widths[:-1], widths[1:]):
                add_block(rho_layers, in_d, out_d, add_norm_act_dropout=True)
            # final: widths[-1] -> 1 (no LN/act/dropout)
            add_block(rho_layers, widths[-1], 1, add_norm_act_dropout=False)

        self.rho = nn.Sequential(*rho_layers)
        
    def forward(self, dataset, batch_override=None): 
        '''
        dataset is (1 x N x D) matrix, not [N x D]
        '''
        bs = self.batch_size
        #handle embeddings
        #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
        if batch_override:
            bs = batch_override
        x_phi = self.phi(dataset)                
        global_context = x_phi.mean(dim=1, keepdim=True) 
        #global_context = self.hidden_ln(global_context)  
        x_rho = self.rho(x_phi + global_context)          # [N, 1]
        logits = x_rho.squeeze(2)
        scaled_logits = self.logit_scale * logits
        #temp log layer for testing
        #logits = torch.log(logits)
        #probs = F.softmax(x_rho.squeeze(-1), dim=0)       # [N]
        logits_rep = scaled_logits.repeat((self.sample_size*bs,1))
        #matrix = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=1) # give index
        # manual gumbel softmax
        gumbels = -torch.empty_like(logits_rep, memory_format=torch.legacy_contiguous_format).exponential_(generator=self.rngs['torch_cuda']).log()
        # Apply the Gumbel-Softmax transformation
        y = (logits_rep + gumbels) / self.temperature
        
        matrix = y.softmax(dim=-1)
        output = torch.matmul(matrix, dataset)
        output = output.reshape(bs,self.sample_size,dataset.shape[2])
        return output, matrix.detach(), logits
    
    def get_weights(self, dataset):
        self.eval()
        with torch.no_grad():
            #x = embed_data(self.embedding_layers,self.embedding_dict,dataset.unsqueeze(0))
            x_phi = self.phi(dataset)                     # [N, hidden_dim]
            global_context = x_phi.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            global_context = self.hidden_ln(global_context)  
            x_rho = self.rho(x_phi + global_context)#.squeeze().unsqueeze(0)            # [1, N]
            output = F.softmax(x_rho, dim=1)       # [N]
        return output.cpu().squeeze(0).numpy().T
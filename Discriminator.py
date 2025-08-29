import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import sys, os
sys.path.append(os.path.dirname(__file__))
from util import warmup_spectral_norm, create_embedding_layers, embed_data

class DataDiscriminator(torch.nn.Module):
    def __init__(self,
                 subset_size,
                 data_dimension,
                 device,):
        '''
        1. maybe discriminator is too weak (in terms of expressability)
        2. generator should be slowed down in terms of the discriminator
        3. look at tabular GAN papers for how they structure discriminator
        '''
        dropout = 0.1
        super(DataDiscriminator, self).__init__()
        self.num_hidden = 256
        self.model = torch.nn.Sequential(
            nn.Linear(in_features=data_dimension*subset_size, out_features=self.num_hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.num_hidden, 1),
            #nn.LeakyReLU(),
        )
        self.model.to(device)

    def forward(self, input):
        output = self.model(input)
        return output
    
class Discriminator(nn.Module):
    def __init__(self,
                 seed,
                 num_features,
                 dropout,
                 layers):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #self.linear = nn.Linear(num_features, 1)
        modules = []
        if layers is None:
            modules.append(nn.Linear(num_features, 1))
        else:
            modules.append(nn.Linear(num_features,layers[0]))
            modules.append(nn.LayerNorm(layers[0]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(dropout))
            for i,l in enumerate(layers):
                if i == 0:
                    continue
                modules.append(nn.Linear(layers[i-1],l))
                modules.append(nn.LayerNorm(l))
                modules.append(nn.LeakyReLU(0.2))
                modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(layers[-1],1))
            modules.append(nn.Identity())
        self.model = nn.Sequential(*modules)
        self.apply_he_init_to_sequential(self.model)

    def apply_he_init_to_sequential(self,model):
        for layer_id, layer in enumerate(model):
            #initialize all intermediary layers using relu non linearity
            if isinstance(layer, torch.nn.Linear):
                if isinstance(model[layer_id+1],torch.nn.LeakyReLU):  # Apply He initialization to Linear layers
                    torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')  # or kaiming_uniform_
                elif isinstance(model[layer_id+1],torch.nn.Identity):
                    torch.nn.init.xavier_uniform_(layer.weight) 
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)  # Initialize biases to 0

    def forward(self, input):
        input = input.reshape(input.shape[0],input.shape[1]*input.shape[2])
        output = self.model(input)
        return output

class DeepSetCritic(nn.Module):
    def __init__(self, 
                 rngs,
                 num_features,
                 dropout,
                 layers,
                 embedding_dict,
                 hidden_dim=1024,):
        super().__init__()
        self.embedding_dict = embedding_dict
        self.embedding_layers = create_embedding_layers(self.embedding_dict,torch.device('cuda:0'))
        
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

    def init_bias_only(self,model):
        for layer_id, layer in enumerate(model):
            if isinstance(layer, torch.nn.Linear):
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias) 

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

    def forward(self, x):  # x: [N, 8]
        #embed data
        #x = embed_data(self.embedding_layers,self.embedding_dict,x)
        # x: [batch_size, set_size, input_dim]
        x = self.phi(x)         # [B, 32, H]
        x = x.mean(dim=1)        # [B, H] — permutation-invariant pooling
        out = self.rho(x)       # [B, 1]
        return out
    
    def debugging_forward(self, x):  # x: [N, 8]
        # x: [batch_size, set_size, input_dim]
        # x: [batch_size, set_size, input_dim]
        print("x input", x.mean().item(), x.std().item())
        x = self.phi(x)
        print("phi output mean/std/max:", x.mean().item(), x.std().item(), x.abs().max().item())
        print(x.shape)
        x = x.mean(dim=1)
        print("after mean pooling:", x.mean().item(), x.std().item(), x.abs().max().item())
        print(x.shape)
        out = self.rho(x)
        print("final output:", out.mean().item(), out.std().item(), out.abs().max().item())
        print(out.shape)
        exit(1)
        return out

    def set_eval(self):
        self.phi.eval()
        self.rho.eval()
    
    def set_train(self):
        self.phi.train()
        self.rho.train()
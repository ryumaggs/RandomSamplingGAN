import torch
import torch.nn as nn

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
                 num_features):
        super(Discriminator, self).__init__()
        self.num_hidden = 512
        self.linear1 = nn.Linear(num_features, self.num_hidden)
        self.dropout1 = nn.Dropout(0.2)
        self.activation1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.dropout2 = nn.Dropout(0.2)
        self.activation2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(self.num_hidden, 1)
        self.dropout3 = nn.Dropout(0.2)
        self.activation3 = nn.LeakyReLU()

    def forward(self, input):
        output = self.linear1(input)
        output = self.dropout1(output)
        #output = self.activation1(output)
        output = self.linear2(output)
        output = self.dropout2(output)
        #utput = self.activation2(output)
        output = self.linear3(output)
        output = self.dropout3(output)
        #utput = self.activation3(output)
        return output

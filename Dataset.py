import numpy as np
import random
import csv
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
import torch
from util import dict2vector
from DataProcessing import *
from sklearn.preprocessing import StandardScaler

class XBoxDatasetSimulation():
    def __init__(self,
                 dataset_path=None,
                 device=None,):
        self.type = "real"
        self.device = device
        self.df = self.load_csv(dataset_path)

        #logic to split the df
    
    def clean_AGE(self, df):
        #PUTS AGE INTO BINS TO MIMIC XBOX
        #bins = [(18,(30,44):2,(45,64):3,(65,200):4]
        bins = [18,30,45,65,200]
        # Let us create our labels:
        labels = [1, 2, 3, 4]
        # Finally, we add a new column to the df:
        df['AGE'] = pd.cut(df['AGE'], bins=bins, labels=labels)
        df.dropna(inplace=True) #drops any samples with ages outside range
        df['AGE'] = df['AGE'].astype(int)

    def clean_RACE(self, df):
        #PUTS RACE INTO BINS TO MIMIC XBOX
        #bins = [(18,(30,44):2,(45,64):3,(65,200):4]
        bins = [1,2,3,20]
        # Let us create our labels:
        labels = [1,2,3]
        # Finally, we add a new column to the df:
        df['RACE'] = pd.cut(df['RACE'], bins=bins, labels=labels)
        df.dropna(inplace=True) #drops any samples with ages outside range
        df['RACE'] = df['RACE'].astype(int)

    def clean_EDUC(self, df):
        #< HS: 0-5, HS: 6, some_college: 7-9, finished college, 10-11
        #GT_education = {'< HS': 0.05, 'HS':0.15, 'Some college': 0.3, 'finished college': 0.5}

        #PUTS RACE INTO BINS TO MIMIC XBOX
        #bins = [(18,(30,44):2,(45,64):3,(65,200):4]
        bins = [0,6,7,10,30]
        # Let us create our labels:
        labels = [1,2,3,4]
        # Finally, we add a new column to the df:
        df['EDUC'] = pd.cut(df['EDUC'], bins=bins, labels=labels)
        df.dropna(inplace=True) #drops any samples with ages outside range
        df['EDUC'] = df['EDUC'].astype(int)

    def load_csv(self, ground_truth_path):
        try:
            df = pd.read_csv(ground_truth_path)
            df = clean_NaN_target_col(df)
            NaN_cols_list = df.columns[df.isna().any()].tolist()
            clean_NaN_by_col_name(df, NaN_cols_list)
        except Exception as e:
            print(e)
            print('error occured')
            exit(1)
        print(df.shape)
        self.clean_AGE(df)
        self.clean_RACE(df)
        self.clean_EDUC(df)
        print(df.shape)
        return df

class Axios_ipsosdataset():
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 device=None,
                 gt_limit = 5000,):
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit
        self.biased_dataset = self.load_csv(bias_path).to_numpy(dtype=np.float, na_value=0)
        raw_gt_load = self.load_csv(ground_truth_path)
        self.biased_labels = self.biased_dataset[:,-1]
        self.scaler = StandardScaler()
        self.ground_truth_dataset = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT']).to_numpy(dtype=np.float, na_value=0)
        del raw_gt_load
        self.scaler = self.scaler.fit(self.ground_truth_dataset[:self.gt_limit,1:])
        self.ground_truth_dataset = self.scaler.transform(self.ground_truth_dataset[:self.gt_limit,1:])
        self.biased_dataset = self.scaler.transform(self.biased_dataset[:,1:-1])

        self.ground_truth = None
        self.ground_truth_demographics = None
        #bias has axios weights in column 0 and vaccinated status in column -1
        #gt has census weights in column 0
        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset,device=self.device,dtype=torch.float32)
        self.biased_dataset = torch.tensor(self.biased_dataset,device=self.device,dtype=torch.float32)
        

    def load_csv(self, ground_truth_path):
        try:
            df = pd.read_csv(ground_truth_path)
        except Exception as e:
            print(e)
            print('error occured')
            exit(1)
        
        return df
class RealPredictionDataset():
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 device=None,):
        self.type = "real"
        self.device = device
        self.biased_dataset = self.load_csv(bias_path)
        self.ground_truth_dataset = self.load_csv(ground_truth_path)
        self.biased_dataset = torch.tensor(self.biased_dataset,dtype=torch.float32).to(self.device)
        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset,dtype=torch.float32).to(self.device)
        self.ground_truth = torch.mean(self.ground_truth_dataset[:,-1]).cpu().numpy()

        self.ground_truth_dataset = self.ground_truth_dataset[:,:-1]
        self.biased_labels = self.biased_dataset[:,-1]
        self.biased_dataset = self.biased_dataset[:,:-1]
        
        self.ground_truth_demographics = torch.mean(self.ground_truth_dataset,dim=0).cpu().numpy()
    
    def load_csv(self, ground_truth_path):
        try:
            df = pd.read_csv(ground_truth_path)
        except Exception as e:
            print(e)
            print('error occured')
            exit(1)
        
        return df.to_numpy(dtype=np.float, na_value=0)

class RealImportanceDataset():
    def __init__(self,
                 ground_truth_path,
                 num_biased_data_points=None,
                 num_totaldata_ground_truth=None,
                 device=None,):
        self.type = "importance"
        self.device = device
        self.biased_dataset = self.load_csv(ground_truth_path)
        self.create_random_ground_truth() 
        self.ground_truth_dataset = np.zeros((num_totaldata_ground_truth,self.biased_dataset.shape[1]))
        for i in range(num_totaldata_ground_truth):
            sampled_index = np.random.choice(np.arange(self.biased_dataset.shape[0]),p=self.ground_truth)
            self.ground_truth_dataset[i] = self.biased_dataset[sampled_index]
        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset, 
                                                 dtype=torch.float32).to(device)
        self.biased_dataset = torch.tensor(self.biased_dataset,dtype=torch.float32).to(self.device)
    
    def load_csv(self, ground_truth_path):
        try:
            df = pd.read_csv(ground_truth_path)
        except:
            print('error occured')
            exit(1)
        
        return df.to_numpy(dtype=np.float, na_value=0)
    
    def create_random_ground_truth(self):
        random_importance_vector = np.random.random(size=(22,1))
        self.ground_truth = (self.biased_dataset @ random_importance_vector).flatten()
        self.ground_truth = [i / sum(self.ground_truth) for i in self.ground_truth]


class ImportanceDataset():
    def __init__(self,
                 ground_truth_path=None,
                 num_biased_data_points=10,
                 device=None):
        self.type = "importance"
        self.feature_names = []
        self.num_biased_data_points = num_biased_data_points
        self.ground_truth_path = ground_truth_path

        self.zero_prob = 0
        self.num_totaldata_ground_truth = 1000
        attributes = ["Age", "Height", "Gender", "Residency", "Homeland"]

        self.biased_data = []
        self.importance = self.construct_importance(attributes)
        for i in range(num_biased_data_points):
            data = self.construct_datapoint(attributes)
            self.biased_data.append(data)

        self.logits = []
        for datapoint in self.biased_data:
            logit = 0
            for key, value in datapoint.items():
                if type(value) == list:
                    logit += sum([i*j for i,j in zip(value, self.importance[key])])
                else:
                    logit += self.importance[key] * value
            self.logits.append(logit)
        #print("Their logits = Data * Importance (With probability", self.zero_prob, "to be zeroed):")
        #print(self.logits)
        #print("Mark: Ground truth dataset will sample according to these logits. ")
        #print()

        self.probs_theory = [i / sum(self.logits) for i in self.logits]
        self.ground_truth_dataset = np.random.choice(self.biased_data, self.num_totaldata_ground_truth, p=self.probs_theory)
        #print("Ground truth dataset contains", self.num_totaldata_ground_truth, "datapoints. ")
        #print("First 10 data looks like:")
        #print(self.ground_truth_dataset[0:10])
        #print()

        self.ground_truth = [0 for _ in self.biased_data]
        for data in self.ground_truth_dataset:
            if data in self.biased_data:
                self.ground_truth[self.biased_data.index(data)] += 1
        self.ground_truth = [i / self.num_totaldata_ground_truth for i in self.ground_truth]
        self.device = device
        self.ground_truth_dataset = torch.tensor(dict2vector(self.ground_truth_dataset), dtype=torch.float32).to(self.device)
        self.biased_dataset = torch.tensor(dict2vector(self.biased_data), dtype=torch.float32).to(self.device)


    def construct_datapoint(self, attributes):
        datapoint = {}
        if "Age" in attributes:
            datapoint["Age"] = np.random.randint(0,100)/100
        if "Height" in attributes:
            datapoint["Height"] = round(np.random.uniform(0,2), 2)/2
        if "Gender" in attributes:
            classes = [0, 1]
            probabilities = [0.5, 0.5]
            random_class = np.random.choice(classes, p=probabilities)
            result = [0,0]
            result[random_class] = 1
            datapoint["Gender"] = result
        if "Residency" in attributes:
            catogrory = np.random.randint(0, 2)
            result = [0,0,0]
            result[catogrory] = 1
            datapoint["Residency"] = result
        if "Homeland" in attributes:
            catogrory = np.random.randint(0, 3)
            result = [0,0,0,0]
            result[catogrory] = 1
            datapoint["Homeland"] = result

        return datapoint

    def construct_importance(self,attributes):
        importance = {}
        if "Age" in attributes:
            importance["Age"] = round(np.random.uniform(-1, -0.5), 2)
        if "Height" in attributes:
            importance["Height"] = round(np.random.uniform(0, 1), 3)
        if "Gender" in attributes:
            importance["Gender"] = [round(np.clip(np.random.normal(0.75, 0.1), 0, 1), 2), round(np.clip(np.random.normal(0.25, 0.1), 0, 1), 2)]
        if "Residency" in attributes:
            importance["Residency"] = [np.clip(np.random.normal(0.5, 0.1), 0, 1), np.clip(np.random.normal(0.5, 0.1), 0, 1), np.clip(np.random.normal(0, 0.1), 0, 1)]
        if "Homeland" in attributes:
            importance["Homeland"] = [round(item, 2) for item in np.random.uniform(0, 1, [1,4]).tolist()[0]]
        return importance
    
class DataSet():
    def __init__(self,
                 ground_truth_path=None,
                 sample_size=5,
                 batch_size=1,
                 max_data_size=10000):
        self.feature_names = []
        self.max_data_size = max_data_size
        self.sample_size=sample_size
        self.batch_size=batch_size
        self.ground_truth_path = ground_truth_path
    
    def identity_matrix_dataset(self):
        self.groundtruth_data = np.eye(10)
        self.n_truth = self.groundtruth_data.shape[0]
        self.groundtruth_weights =np.zeros(self.n_truth)
        self.biased_data = self.groundtruth_data
        self.n_biased = self.biased_data.shape[0]
        self.x_dim = self.groundtruth_data.shape[1]
        self.biased_weights = np.zeros(self.n_biased)
        
        for i in range(self.groundtruth_data.shape[0]):
            self.groundtruth_weights[i] = 0
            self.biased_weights[i] = 1/self.x_dim
        self.groundtruth_weights[0] = 0.4
        self.groundtruth_weights[1] = 0.3
        self.groundtruth_weights[2] = 0.2
        self.groundtruth_weights[3] = 0.1
    
    def toy_dataset(self):
        groundtruth_data = ["Male, Native" for _ in range(66)] + ["Female, Native" for _ in range(66)] + ["Male, Foreign" for _ in range(66)] + ["Female, Foreign" for _ in range(0)]
        random.shuffle(groundtruth_data)
        self.groundtruth_data = np.zeros((66*3,2))
        self.groundtruth_weights = np.zeros(self.groundtruth_data.shape[0])
        for i,dp in enumerate(groundtruth_data):
            if "Female" in dp:
                self.groundtruth_data[i][0] = 0
            else:
                self.groundtruth_data[i][0] = 1
            if "Native" in dp:
                self.groundtruth_data[i][1] = 0
            else:
                self.groundtruth_data[i][1] = 1
            self.groundtruth_weights[i] = 1/(self.groundtruth_data.shape[0])
        self.n_truth = self.groundtruth_data.shape[0]
        biased_data = ["Male, Native" for _ in range(1)] + ["Female, Native" for _ in range(1)] + ["Male, Foreign" for _ in range(1)] + ["Female, Foreign" for _ in range(1)]
        self.biased_data = np.zeros((4,2))
        self.biased_weights = np.zeros(self.biased_data.shape[0])
        for i,dp in enumerate(biased_data):
            if "Female" in dp:
                self.biased_data[i][0] = 0
            else:
                self.biased_data[i][0] = 1
            if "Native" in dp:
                self.biased_data[i][1] = 0
            else:
                self.biased_data[i][1] = 1
            self.biased_weights[i] = 1/(self.biased_data.shape[0])
        self.n_biased = self.biased_data.shape[0]
        #random.shuffle(bias_dataset)

    def csv_dataset(self):
        if self.ground_truth_path is not None:
            self.groundtruth_data = self.load_csv(self.ground_truth_path)
            self.n_truth = self.groundtruth_data.shape[0]
            self.groundtruth_weights = np.arange(self.n_truth)/self.n_truth
            self.groundtruth_weights = softmax(self.groundtruth_weights)
            self.biased_data = self.groundtruth_data
            self.n_biased = self.biased_data.shape[0]
            self.biased_weights = np.zeros(self.n_biased)
            self.x_dim = self.groundtruth_data.shape[1]

    def load_csv(self,data_path):
        with open(data_path, newline='\n') as csvfile:
            data_count = 0
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            data = []
            for row_count, row in tqdm(enumerate(spamreader)):
                if row_count == 0:
                    self.feature_names = row
                else:
                    cur_data = np.zeros(len(row),dtype=float)
                    for did, datum in enumerate(row):
                        if datum == '':
                            cur_data[did] = -1.0
                        else:
                            cur_data[did] = float(datum)
                    data.append(cur_data)
                    data_count += 1
                    if data_count >= self.max_data_size:
                        break
        data = np.array(data)
        return data

    def get_groundtruth_batch(self):
        batch = []
        for _ in range(self.batch_size): 
            groundtruth_indexes = np.random.choice(a=self.n_truth,size=self.sample_size,p=self.groundtruth_weights)
            batch.append(self.groundtruth_data[groundtruth_indexes].flatten())
        batch = np.stack(batch)
        return batch
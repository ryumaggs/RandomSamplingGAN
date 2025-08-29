import numpy as np
import random
import csv
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
import torch
from util import dict2vector, normalize_to_minus1_plus1, embed_data
from DataProcessing import *
from sklearn.preprocessing import StandardScaler
from HouseholdCensusDataProcessing import *
from matplotlib import pyplot as plt

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

class HouseholdPulse_dataset():
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 rngs,
                 num_labels=1,
                 device=None,
                 columns_to_keep = None,
                 gt_limit = 5000,
                 bias_limit = 1000, ):
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit
        self.bias_limit = bias_limit
        self.num_labels = num_labels
        raw_bias_load = self.load_csv(bias_path)#.to_numpy(dtype=np.float, na_value=0)
        raw_gt_load = self.load_csv(ground_truth_path)
        self.embedding_dict = self.create_embedding_dictionary(raw_gt_load)
        self.num_categories_per_features = [raw_gt_load[col].nunique() for col in raw_gt_load.columns]
        self.num_categories_per_features = self.num_categories_per_features[1:]
        
        if type(rngs) == list:
            gt_list = []
            bd_list = []
            for rng in rngs:
                gtd, bd = self.load_data_with_rngs(columns_to_keep,
                                                    raw_bias_load,
                                                    self.bias_limit,
                                                    raw_gt_load,
                                                    self.gt_limit,
                                                    rng)
                bd_list.append(bd)
                gt_list.append(gtd)
            self.ground_truth_dataset = np.concatenate(gt_list, axis=0)
            self.biased_dataset = np.concatenate(bd_list, axis=0)
        else:
            self.ground_truth_dataset, self.biased_dataset = self.load_data_with_rngs(
                                                                columns_to_keep,
                                                                raw_bias_load,
                                                                bias_limit,
                                                                raw_gt_load,
                                                                gt_limit,
                                                                rngs)
        del raw_gt_load
        del raw_bias_load 
        self.synthetic_label_1 = None
        self.synthetic_label_2 = None
        self.var_setup()
    
    def create_embedding_dictionary(self,df):
        unique_counts = df.nunique().to_dict()
        del unique_counts['PERWT']
        uc = {}
        for i, name in enumerate(unique_counts):
            if unique_counts[name] <= 20:
                uc[i] = ["onehot", unique_counts[name], unique_counts[name]]
            else:
                uc[i] = ["embed", unique_counts[name], min(50, int((unique_counts[name]+1) **0.5))]
        return uc
        
    def var_setup_onhold(self):
        self.biased_labels = self.biased_dataset[:,0]
        print("uniform avg target: ", sum(self.biased_labels)/len(self.biased_labels))
        self.unscaled_ground_truth = self.ground_truth_dataset[:,1:]
        self.unscaled_biased = self.biased_dataset[:,1:]

        self.ground_truth_dataset, self.biased_dataset = normalize_to_minus1_plus1(self.unscaled_ground_truth, self.unscaled_biased)

        self.ground_truth = None
        self.ground_truth_demographics = None

        #bias has axios weights in column 0 and vaccinated status in column -1
        #gt has census weights in column 0
        self.gt_cpu = self.ground_truth_dataset
        self.bias_cpu = self.biased_dataset

        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset,device=self.device,dtype=torch.float32)
        self.biased_dataset = torch.tensor(self.biased_dataset,device=self.device,dtype=torch.float32)

    def var_setup(self):
        self.biased_labels = self.biased_dataset[:,:self.num_labels]
        for nl in range(self.num_labels):
            print("uniform avg target: ", sum(self.biased_labels[:,nl])/len(self.biased_labels[:,nl]))
        self.unscaled_ground_truth = self.ground_truth_dataset[:,1:]
        self.unscaled_biased = self.biased_dataset[:,1:]

        self.ground_truth = None
        self.ground_truth_demographics = None
        
        self.gt_cpu = self.ground_truth_dataset
        self.bias_cpu = self.biased_dataset

        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset[:,1:],device=self.device,dtype=torch.float32)
        self.biased_dataset = torch.tensor(self.biased_dataset[:,1:],device=self.device,dtype=torch.float32)

        self.ground_truth_dataset = embed_data(None,self.embedding_dict,self.ground_truth_dataset)
        self.biased_dataset = embed_data(None,self.embedding_dict,self.biased_dataset).unsqueeze(0)

    def compute_bin_centers_DEPRICATED(self, scaler, num_bins_per_feature, device="cpu"):
        """
        Compute standardized bin centers for categorical features whose original bin values start at 1.

        Parameters:
            scaler: fitted sklearn StandardScaler (with .mean_ and .scale_)
            num_bins_per_feature: list of ints (length = number of features)
            device: 'cpu' or 'cuda'

        Returns:
            bin_centers: list of torch tensors, one per feature (each of shape (K_f,))
        """
        means = scaler.mean_     # shape: (9,)
        scales = scaler.scale_   # shape: (9,)

        bin_centers = []

        for f in range(len(num_bins_per_feature)):
            K = num_bins_per_feature[f]                          # Number of bins for feature f
            bin_ids = np.arange(1, K + 1)                        # Bin indices: [1, 2, ..., K]
            standardized = (bin_ids - means[f]) / scales[f]      # Standardize each bin center
            bin_centers.append(torch.tensor(standardized, dtype=torch.float32, device=device))

        return bin_centers

    def load_data_with_rngs(self, 
                            columns_to_keep,
                            raw_bias_load,
                            bias_limit,
                            raw_gt_load,
                            gt_limit,
                            rngs):
        if columns_to_keep is not None:
            census_columns_to_keep = ['PERWT']
            census_columns_to_keep.extend(list(columns_to_keep))
            gtd = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT'],random_state=rngs['seed_gt'])[census_columns_to_keep].to_numpy(dtype=np.float, na_value=0)
            survey_columns_to_keep = ['RECVDVACC']
            survey_columns_to_keep.extend(list(columns_to_keep))
            if bias_limit >= raw_bias_load.shape[0]:
                bd = raw_bias_load[survey_columns_to_keep].to_numpy(dtype=np.float, na_value=0)
            else:
                bd = raw_bias_load.sample(n=bias_limit,random_state=rngs['seed_bias'])[survey_columns_to_keep].to_numpy(dtype=np.float, na_value=0)

        else:
            gtd = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT'],random_state=rngs['seed_gt']).to_numpy(dtype=np.float, na_value=0)
            if bias_limit > raw_bias_load.shape[0]:
                bd = raw_bias_load.to_numpy(dtype=np.float, na_value=0)
            else:
                bd = raw_bias_load.sample(n=bias_limit,random_state=rngs['seed_bias']).to_numpy(dtype=np.float, na_value=0)
        return gtd, bd
    
    def create_combination_id_mapping(self,np1, np2):
        #convert both to pandas df
        df1 = pd.DataFrame(np1)
        df2 = pd.DataFrame(np2)

        combos_df1 = df1.apply(tuple, axis=1)
        combos_df2 = df2.apply(tuple, axis=1)

        # Factorize to get unique IDs
        ids, uniques = pd.factorize(combos_df1)
        ids, uniques2 = pd.factorize(combos_df2)
        
        # Store mapping as a dictionary
        combo_to_id = {combo: idx for idx, combo in enumerate(uniques)}
        for combo in uniques2:
            if combo not in combo_to_id:
                combo_to_id[combo] = len(combo_to_id)
        #apply mapping
        df1['combo_id'] = combos_df1.map(combo_to_id)
        df2['combo_id'] = combos_df2.map(combo_to_id)
        #counting unique combinations
        set1 = set(df1['combo_id'].unique())
        set2 = set(df2['combo_id'].unique())

        # Values unique to each DataFrame
        only_in_df1 = set1 - set2
        only_in_df2 = set2 - set1
        print("Values only in df1:", len(only_in_df1))
        print("Values only in df2:", len(only_in_df2))

        #convert back to numpy
        return df1.to_numpy(), df2.to_numpy()

    def load_csv(self, ground_truth_path):
        try:
            df = pd.read_csv(ground_truth_path)
        except Exception as e:
            print(e)
            print('error occured')
            exit(1)
        
        return df

class HouseholdPulse_synthetic(HouseholdPulse_dataset):
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 rngs,
                 device=None,
                 gt_limit = 5000,
                 bias_limit = 1000, ):
        '''
        column_names - list[str] - len(2) - 2 valid column names in the ground_truth dataset 
        '''
        self.num_labels = 0
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit
        self.bias_limit = bias_limit
        rng = rngs
        raw_gt_load = self.load_csv(ground_truth_path)
        self.ground_truth_dataset = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT'],random_state=rng['seed_gt'])
        self.embedding_dict = self.create_embedding_dictionary(raw_gt_load)
        self.num_categories_per_features = [raw_gt_load[col].nunique() for col in raw_gt_load.columns]
        self.num_categories_per_features = self.num_categories_per_features[1:]

        #pick two random variables
        all_columns = list(self.ground_truth_dataset.columns)
        indexes = np.arange(len(all_columns))[1:]
        var_1_index = np.random.choice(indexes)
        indexes = indexes[indexes != var_1_index]
        var_2_index = np.random.choice(indexes)
        var_1 = all_columns[var_1_index]
        var_2 = all_columns[var_2_index]
        self.column_names = [var_1, var_2] #store selected columns for use at :355
        self.col_indexes = [var_1_index-1, var_2_index-1]

        #compute and store original joint distribution
        joint_dist = pd.crosstab(self.ground_truth_dataset[var_1], self.ground_truth_dataset[var_2], normalize=True)
        joint_np = joint_dist.values
        x_vals = sorted(self.ground_truth_dataset[var_1].unique())
        y_vals = sorted(self.ground_truth_dataset[var_2].unique())
        self.original_distribution = np.copy(joint_np).flatten()
        self.var_counts = [len(x_vals),len(y_vals)]

        #pick random cell in the joint distribution to be upscaled
        rand_x = np.random.choice(np.arange(len(x_vals)))
        rand_y = np.random.choice(np.arange(len(y_vals)))
        self.upscaled_cell = [rand_x, rand_y]

        #Resample new joint after upscaling cell by epsilon
        p = joint_np[rand_x, rand_y]
        # Random perturbation: about ±25–50% of the cell's current value
        epsilon = np.random.uniform(0.5,1)
        #epsilon = 0.5
        joint_np[rand_x, rand_y] += epsilon
        joint_np = np.clip(joint_np, 0.01, None)   # keep non-negative
        joint_np /= joint_np.sum()  # Renormalize
        new_joint_distr = self.resample_df_with_joint_distribution(self.column_names, self.ground_truth_dataset, joint_np, x_vals, y_vals, bias_limit)
        #experimental
        self.missing_values = self.has_missing_joints(new_joint_distr, col_i=var_1, col_j=var_2)

        #determining what is missing from data set
        self.biased_dataset = new_joint_distr.to_numpy(dtype=np.float, na_value=0)
        self.ground_truth_dataset = self.ground_truth_dataset.to_numpy(dtype=np.float, na_value=0)

        self.var_setup()
        self.add_random_noise()

    def add_random_noise(self):
        categories_list = [np.random.randint(2,9) for _ in range(len(self.embedding_dict))]
        #add random noise and augments embedding dict
        random_gt = self.create_random_categories(self.unscaled_ground_truth.shape[0],
                                                  categories_list=categories_list,
                                                  m = 'gt')
        random_bias = self.create_random_categories(self.unscaled_biased.shape[0],
                                                    categories_list=categories_list,
                                                    m = 'bias',
                                                    add_to_embedding_dict=False).unsqueeze(0)
        self.ground_truth_dataset = torch.concat((self.ground_truth_dataset,random_gt.to(self.device)),dim=-1)
        self.biased_dataset = torch.concat((self.biased_dataset,random_bias.to(self.device)),dim=-1)

    def create_random_categories(self,num_points,
                                 categories_list,
                                 m,
                                 add_to_embedding_dict=True):
        new_features = []
        locked_len = len(self.embedding_dict)
        for new_i in range(len(categories_list)):
            num_categories = categories_list[new_i]   # between 2 and 8 inclusive
            new_col = np.random.randint(0, num_categories, size=num_points)
            if m == 'gt':
                self.unscaled_ground_truth = np.column_stack((self.unscaled_ground_truth,new_col))
            else:
                self.unscaled_biased = np.column_stack((self.unscaled_biased,new_col))
            new_features.append(new_col)
            if add_to_embedding_dict:
                self.embedding_dict[new_i+locked_len] = ['onehot', num_categories, num_categories]
        new_features = torch.tensor(np.column_stack(new_features))
        if add_to_embedding_dict:
            new_features = embed_data(None,self.embedding_dict,new_features,overrite_start_idx=locked_len)
        else:
            new_features = embed_data(None,self.embedding_dict,new_features,overrite_start_idx=locked_len//2)

        return new_features

    def has_missing_joints(self, df, col_i='col_i', col_j='col_j'):
        unique_i = df[col_i].unique()
        unique_j = df[col_j].unique()

        full_index = pd.MultiIndex.from_product([unique_i, unique_j])
        observed_index = pd.MultiIndex.from_frame(df[[col_i, col_j]].drop_duplicates())

        return not full_index.isin(observed_index).all()

    def resample_df_with_joint_distribution(self,col_names, df, joint_np, x_vals, y_vals, n_samples):
        # Create DataFrame for desired (X, Y) frequencies
        xy_freq = pd.DataFrame(
            [(x_vals[i], y_vals[j], joint_np[i, j]) for i in range(len(x_vals)) for j in range(len(y_vals))],
            columns=[col_names[0], col_names[1], 'prob']
        )

        # Compute how many samples to take from each (X, Y) group
        xy_freq['n'] = (xy_freq['prob'] * n_samples).round().astype(int)

        resampled_parts = []

        for _, row in xy_freq.iterrows():
            x_val, y_val, n = row[col_names[0]], row[col_names[1]], row['n']
            group = df[(df[col_names[0]] == x_val) & (df[col_names[1]] == y_val)]

            if len(group) == 0:
                continue  # Skip if no such group exists in original data

            # Sample with replacement if necessary
            n = int(n)
            sampled = group.sample(n=n, replace=(n > len(group)))
            resampled_parts.append(sampled)

        # Combine all sampled parts
        resampled_df = pd.concat(resampled_parts, ignore_index=True)
        
        return resampled_df

    def sample_categorical_distribution(self,df, column, target_dist, K, replace=False, random_state=None):
        """
        Sample K rows from df such that the distribution of the values in the specified column
        matches the target categorical distribution.

        Parameters:
            df (pd.DataFrame): Original DataFrame.
            column_index (int): Index of the column to match distribution on.
            target_dist (dict): Target distribution (e.g., {0: 0.5, 1: 0.3, 2: 0.2}).
            K (int): Total number of samples to draw.
            replace (bool): Whether to sample with replacement.
            random_state (int or None): Seed for reproducibility.

        Returns:
            pd.DataFrame: Sampled DataFrame of size K.
        """
        result_dfs = []
        
        for category, proportion in target_dist.items():
            num_samples = int(round(proportion * K))
            subset = df[df[column] == category]
            
            if len(subset) == 0:
                raise ValueError(f"No samples found for category '{category}' in the specified column.")
            if not replace and num_samples > len(subset):
                raise ValueError(f"Not enough samples in category '{category}' to sample {num_samples} without replacement.")
            
            sampled = subset.sample(n=num_samples, replace=replace, random_state=random_state)
            result_dfs.append(sampled)
        
        result = pd.concat(result_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Fix any rounding issues (e.g., if total != K due to rounding)
        if len(result) > K:
            result = result.sample(n=K, random_state=random_state)
        elif len(result) < K:
            extra = df.sample(n=K - len(result), replace=replace, random_state=random_state)
            result = pd.concat([result, extra]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        return result

class Ari_dataset(HouseholdPulse_dataset):
    def dummy(self):
        pass

class D4P_dataset():
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 rngs,
                 device=None,
                 gt_limit = 5000, ):
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit

        raw_bias_load = self.load_csv(bias_path)#.to_numpy(dtype=np.float, na_value=0)
        raw_gt_load = self.load_csv(ground_truth_path)
        #sample ground truth by gt limit
        '''
        gt columns = ['PERWT', 'REGION', 'EDUC', 'INCTOT', 'SEX', 'MARST', 'FAMSIZE', 'RACE',
        'AGE', 'BIDENPERC'],
        '''
        columns_to_keep = list(raw_bias_load.columns)[1:]
        self.biased_dataset = raw_bias_load.to_numpy(dtype=np.float, na_value=0)
        if columns_to_keep is not None:
            census_columns_to_keep = ['PERWT']
            census_columns_to_keep.extend(list(columns_to_keep))
            self.ground_truth_dataset = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT'],random_state=rngs['seed_gt'])[census_columns_to_keep].to_numpy(dtype=np.float, na_value=0)
        del raw_gt_load
        del raw_bias_load 

        self.biased_labels = self.biased_dataset[:,0]
        print("uniform avg vaccine: ", sum(self.biased_labels)/len(self.biased_labels))
        self.unscaled_ground_truth = self.ground_truth_dataset[:,1:]
        self.unscaled_biased = self.biased_dataset[:,1:]
        test = np.concatenate((self.ground_truth_dataset[:,1:],self.biased_dataset[:,1:]))
        self.scaler = StandardScaler()        
        #self.scaler = self.scaler.fit(self.ground_truth_dataset[:,1:])
        self.scaler = self.scaler.fit(test)
        self.ground_truth_dataset = self.scaler.transform(self.ground_truth_dataset[:,1:])
        self.biased_dataset = self.scaler.transform(self.biased_dataset[:,1:])
        self.ground_truth = None
        self.ground_truth_demographics = None

        self.unscaled_ground_truth, self.unscaled_biased = self.create_combination_id_mapping(self.unscaled_ground_truth,
                                                                                                self.unscaled_biased)

        #bias has axios weights in column 0 and vaccinated status in column -1
        #gt has census weights in column 0
        self.gt_cpu = self.ground_truth_dataset
        self.bias_cpu = self.biased_dataset

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
    
    def create_combination_id_mapping(self,np1, np2):
        #convert both to pandas df
        df1 = pd.DataFrame(np1)
        df2 = pd.DataFrame(np2)

        combos_df1 = df1.apply(tuple, axis=1)
        combos_df2 = df2.apply(tuple, axis=1)

        # Factorize to get unique IDs
        ids, uniques = pd.factorize(combos_df1)
        ids, uniques2 = pd.factorize(combos_df2)
        
        # Store mapping as a dictionary
        combo_to_id = {combo: idx for idx, combo in enumerate(uniques)}
        for combo in uniques2:
            if combo not in combo_to_id:
                combo_to_id[combo] = len(combo_to_id)
        #apply mapping
        df1['combo_id'] = combos_df1.map(combo_to_id)
        df2['combo_id'] = combos_df2.map(combo_to_id)
        #counting unique combinations
        set1 = set(df1['combo_id'].unique())
        set2 = set(df2['combo_id'].unique())

        # Values unique to each DataFrame
        only_in_df1 = set1 - set2
        only_in_df2 = set2 - set1
        print("Values only in df1:", len(only_in_df1))
        print("Values only in df2:", len(only_in_df2))

        #convert back to numpy
        return df1.to_numpy(), df2.to_numpy()
    
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

        self.unscaled_ground_truth = self.ground_truth_dataset
        self.unscaled_biased = self.biased_dataset
    
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
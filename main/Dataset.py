#global imports
import numpy as np
import random
import csv
from tqdm import tqdm
from scipy.special import softmax
import pandas as pd
import torch
from DataProcessing import *
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

#local imports
from main.util import dict2vector, normalize_to_minus1_plus1, embed_data
from DataProcessing.HouseholdCensusDataProcessing import *
from typing import Dict, List, Tuple, Any, Optional

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

class HouseholdPulse_dataset():
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 rngs,
                 label_information,
                 device=None,
                 columns_to_keep = None,
                 gt_limit = 5000,
                 bias_limit = 1000, ):
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit
        self.bias_limit = bias_limit
        self.label_names = list(label_information.keys())

        self.label_indexes = list(label_information.values())
        self.num_labels = len(label_information)
        raw_bias_load = self.load_csv(bias_path)#.to_numpy(dtype=np.float, na_value=0)
        raw_gt_load = self.load_csv(ground_truth_path)
        self.embedding_dict = self.create_embedding_dictionary(raw_gt_load)
        self.num_categories_per_features = [raw_gt_load[col].nunique() for col in raw_gt_load.columns]
        self.num_categories_per_features = self.num_categories_per_features[1:]
        self.ground_truth_dataset, self.gt_weights, self.biased_dataset, self.biased_labels = self.load_data_with_rngs(
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
        if 'PERWT' in unique_counts:
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
        self.unscaled_ground_truth = np.copy(self.ground_truth_dataset)
        self.unscaled_biased = np.copy(self.biased_dataset)

        self.ground_truth = None
        self.ground_truth_demographics = None

        self.gt_cpu = self.ground_truth_dataset
        self.bias_cpu = self.biased_dataset

        self.ground_truth_dataset = torch.tensor(self.ground_truth_dataset,device='cpu',dtype=torch.float32)
        self.biased_dataset = torch.tensor(self.biased_dataset,device=self.device,dtype=torch.float32)
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
            raise NotImplementedError
        else:
            gtd = None
            if gt_limit < raw_gt_load.shape[0]:
                gtd = raw_gt_load.sample(n=gt_limit,weights=raw_gt_load['PERWT'],random_state=rngs['seed_gt'])
            else:
                gtd = raw_gt_load
            gtd = gtd.to_numpy(dtype=np.float, na_value=0)
            
            if bias_limit > raw_bias_load.shape[0]:
                bd = raw_bias_load.to_numpy(dtype=np.float, na_value=0)
            else:
                bd = raw_bias_load.sample(n=bias_limit,random_state=rngs['seed_bias'])
                bd = bd.to_numpy(dtype=np.float, na_value=0)

            bl = bd[:,self.label_indexes]
            for nl in range(self.num_labels):
                print("uniform avg target: ", sum(bl[:,nl])/len(bl[:,nl]))
            
            print(gtd.shape, bd.shape, bl.shape)
        return gtd[:,1:], gtd[:,0], bd[:,self.num_labels:], bl
    
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

class Axios_ipsosdataset(HouseholdPulse_dataset):
    def dummy(self):
        pass

class HouseholdPulse_synthetic(HouseholdPulse_dataset):
    def __init__(self,
                 ground_truth_path,
                 bias_path,
                 rngs,
                 label_information,
                 device=None,
                 columns_to_keep = None,
                 gt_limit=5000,
                 bias_limit=1000, 
                 add_random_noise=False,
                 max_num_ran_var=0,):
        '''
        Docstring for __init__
        
        :param self: Description
        :param ground_truth_path: Str - path to census data
        :param bias_path: Str, can be None, dummy holder
        :param rngs: rngs object created in notebook
        :param device: torch.cuda(device) object
        :param columns_to_keep: leave as None, dummy argument
        :param gt_limit: int - number of census points to use
        :param bias_limit: int - number of bias points to use
        :param add_random_noise: bool - should the code add random variables
        :param max_num_ran_var: int - max number of random var, must be multiple of 2
        '''
        self.label_indexes = list(label_information.values())
        self.num_labels = len(label_information)
        self.label_names = list(label_information.keys())
        self.add_random_noise=add_random_noise
        self.max_num_ran_var=max_num_ran_var
        self.num_labels = 0
        self.type = 'real'
        self.device = device
        self.gt_limit = gt_limit
        self.bias_limit = bias_limit
        rng = rngs
        raw_gt_load = self.load_csv(ground_truth_path)
        #raw_gt_load = raw_gt_load.iloc[:,:3]
        self.ground_truth_dataset = raw_gt_load
        self.gt_weights = self.ground_truth_dataset.iloc[:,0].to_numpy()
        self.ground_truth_dataset.drop(self.ground_truth_dataset.columns[0], axis=1, inplace=True)
        if False:
            self.ground_truth_dataset = raw_gt_load.sample(n=gt_limit,
                                                        weights=raw_gt_load['PERWT'],
                                                        random_state=rng['seed_gt'])
            #remove perwt column from both gt and raw
            self.ground_truth_dataset.drop(self.ground_truth_dataset.columns[0], axis=1, inplace=True)
            self.gt_weights = np.ones((self.ground_truth_dataset.shape[0]))

        #cant drop since GT is now unique cells only
        self.embedding_dict = self.create_embedding_dictionary(raw_gt_load)
        self.num_categories_per_features = [raw_gt_load[col].nunique() for col in raw_gt_load.columns]
        self.column_names = []
        self.original_distributions = []
        self.col_indexes = []
        self.upscaled_cells = []
        self.joint_table = {}

        #pick two random variables
        for _ in range(self.max_num_ran_var//2):
            self.pick_perturb_two_vars()
        w, diag = self.fit_weights_to_pairwise_joints(df=raw_gt_load,
                                       pair_targets=self.joint_table,)
        
        #sample with final w
        idx = np.random.choice(raw_gt_load.shape[0], size=bias_limit, replace=True, p=w)
        sampled_df = raw_gt_load.iloc[idx]
        '''
        bl = bd[:,self.label_indexes]
            for nl in range(self.num_labels):
                print("uniform avg target: ", sum(bl[:,nl])/len(bl[:,nl]))
            
            print(gtd.shape, bd.shape, bl.shape)
        return gtd[:,1:], gtd[:,0], bd[:,self.num_labels:], bl
        '''
        #sample data set with distribution and convert to numpy array
        #new_joint_distr = self.resample_df_with_joint_distribution(self.column_names, self.ground_truth_dataset, joint_np, x_vals, y_vals, bias_limit)
        #experimental
        #self.missing_values = self.has_missing_joints(new_joint_distr, col_i=var_1, col_j=var_2)
        self.biased_dataset = sampled_df.to_numpy(dtype=np.float, na_value=0)
        self.ground_truth_dataset = self.ground_truth_dataset.to_numpy(dtype=np.float, na_value=0)
        self.biased_labels = np.ones(self.biased_dataset.shape[0])
        self.var_setup()

        if False: #debbugging up sampling code
            if self.add_random_noise:
                self.add_random_noise()
            w = np.ones(self.unscaled_biased.shape[0])
            i = self.col_indexes[0][0]
            j = self.col_indexes[0][1]
            C_i = int(self.num_categories_per_features[i])
            C_j = int(self.num_categories_per_features[j])

            xi = self.unscaled_biased[:, i].astype(np.int64, copy=False)
            xj = self.unscaled_biased[:, j].astype(np.int64, copy=False)

            # Optional safety checks (can comment out for speed)
            if (xi < 0).any() or (xi >= C_i).any():
                raise ValueError(f"Column {i} has values outside [0, {C_i-1}]")
            if (xj < 0).any() or (xj >= C_j).any():
                raise ValueError(f"Column {j} has values outside [0, {C_j-1}]")

            flat_idx = xi * C_j + xj  # maps (xi,xj) -> [0, C_i*C_j)
            joint = np.bincount(
                flat_idx,
                weights=w,
                minlength=C_i * C_j
            ).reshape(C_i, C_j)


            s = joint.sum()
            if s > 0:
                joint = joint / s
            print(joint)
            assert 1 == 0
        

    def pick_perturb_two_vars(self):
        all_columns = list(self.ground_truth_dataset.columns)
        indexes = np.arange(len(all_columns))
        var_1_index = np.random.choice(indexes)
        indexes = indexes[indexes != var_1_index]
        var_2_index = np.random.choice(indexes)
        var_1 = all_columns[var_1_index]
        var_2 = all_columns[var_2_index]
        self.column_names += [(var_1, var_2)] #store selected columns for use at :355
        self.col_indexes += [(var_1_index, var_2_index)]

        #compute and store original joint distribution
        joint_dist = pd.crosstab(self.ground_truth_dataset[var_1], self.ground_truth_dataset[var_2], normalize=True)
        joint_np = joint_dist.values
        x_vals = sorted(self.ground_truth_dataset[var_1].unique())
        y_vals = sorted(self.ground_truth_dataset[var_2].unique())
        self.original_distributions.append(np.copy(joint_np).flatten())
        self.var_counts = [len(x_vals),len(y_vals)]

        #pick two random variables
        rand_x = np.random.choice(np.arange(len(x_vals)))
        rand_y = np.random.choice(np.arange(len(y_vals)))
        self.upscaled_cells += [(rand_x, rand_y)]

        #Resample new joint after upscaling cell by epsilon
        # Random perturbation: about ±25–50% of the cell's current value
        epsilon = np.random.uniform(0.5,1)
        #epsilon = 0.5
        joint_np[rand_x, rand_y] += epsilon
        joint_np = np.clip(joint_np, 0.01, None)   # keep non-negative
        joint_np /= joint_np.sum()  # Renormalize
        self.joint_table[(var_1_index, var_2_index)] = joint_np

    def reweight_to_joint_targets(self, df, targets, max_iter=1000, tol=1e-6):
        """
        df: original dataset (N x 8)
        pairs: list of (i,j) index pairs
        targets: dict mapping (i,j) -> target joint distribution as DataFrame or np.array
        """
        N = len(df)
        w = np.ones(N) / N  # start with uniform weights
        for it in range(max_iter):
            w_old = w.copy()
            for (i, j), target in targets.items():
                # current weighted joint
                joint = pd.crosstab(df.iloc[:, i], df.iloc[:, j], values=w, aggfunc="sum", normalize=True).to_numpy()
                # ratio of target to current
                ratio = target / (joint + 1e-12)
                
                # map back ratios to individual rows
                mapping = {}
                vals_i = df.iloc[:, i].values
                vals_j = df.iloc[:, j].values
                for ii, vi in enumerate(np.unique(vals_i)):
                    for jj, vj in enumerate(np.unique(vals_j)):
                        mapping[(vi, vj)] = ratio[ii, jj]
                
                # update weights multiplicatively
                w *= np.array([mapping[(vals_i[k], vals_j[k])] for k in range(N)])

            # normalize
            w /= w.sum()
            
            # check convergence
            if np.linalg.norm(w - w_old, 1) < tol:
                break
        
        return w
    
    def fit_weights_to_pairwise_joints(self,
        df: pd.DataFrame,
        pair_targets: List[Dict[Tuple[int, int], np.ndarray]],
        *,
        max_iter: int = 500,
        tol: float = 1e-8,
        damping: float = 1.0,
        eps: float = 1e-12,
        normalize_weights: bool = True,
        return_diagnostics: bool = True,
        verbose: bool = False,
    ) -> Any:
        """
        Compute a weight vector over unique rows of df so that specified pairwise joint
        distributions are matched as closely as possible via iterative proportional fitting (IPF).

        Assumptions:
        - Each row in df is a unique categorical combination (no duplicate rows).
        - df columns are categorical variables encoded as integers.
        - For each constraint (i, j), the target is a 2D array T with shape (C_i, C_j)
            representing desired joint probabilities (or nonnegative mass; we normalize).

        Inputs:
        df: (N, D) dataframe of categorical codes.
        pair_targets: list of dictionaries; each dict: {(i,j): target_2d, ...}
                        You can pass a single dict; list lets you merge multiple sources easily.

        Algorithm:
        - Initialize weights w uniform.
        - For each constraint (i,j):
            compute current joint J_ij from w
            update each row weight multiplicatively by ratio = T[a,b]/J[a,b]
            (optionally damped).
        - Iterate until max absolute joint error across constraints < tol.

        Returns:
        If return_diagnostics:
            (w, diagnostics_dict)
        else:
            w

        Notes:
        - If some target cells have positive mass but df has *no rows* for that (a,b),
            the constraints are infeasible. We detect and report it; IPF cannot create support.
        - If you want least-squares optimal weights under infeasibility, that becomes a
            constrained optimization problem; this IPF approach finds the max-entropy / log-linear
            solution when feasible (and a “best effort” when not, but may stall).
        """
        # ---- Flatten list of dicts into one dict of constraints ----
        constraints: Dict[Tuple[int, int], np.ndarray] = pair_targets
        if len(constraints) == 0:
            raise ValueError("No constraints provided in pair_targets.")

        X = df.to_numpy()
        if X.ndim != 2:
            raise ValueError(f"df must be 2D; got array shape {X.shape}")
        N, D = X.shape

        # Ensure integer-coded
        if not np.issubdtype(X.dtype, np.integer):
            # Try safe conversion if it's categorical/object containing ints
            try:
                X = X.astype(np.int64)
            except Exception as e:
                raise ValueError("df must contain integer-coded categories.") from e

        # ---- Precompute per-constraint row->flat-cell mapping and support masks ----
        precomp = []
        infeasible = []  # store messages for infeasible constraints

        for (i, j), T in constraints.items():
            if not (0 <= i < D and 0 <= j < D):
                raise IndexError(f"Constraint {(i,j)} out of bounds for df with D={D}.")
            if T.ndim != 2:
                raise ValueError(f"Target for {(i,j)} must be 2D; got shape {T.shape}.")
            if (T < 0).any():
                raise ValueError(f"Target for {(i,j)} contains negative entries.")
            Ti = T.copy()
            sT = Ti.sum()
            if sT <= 0:
                raise ValueError(f"Target for {(i,j)} sums to 0; cannot fit.")
            Ti /= sT  # normalize to probability mass

            Ci, Cj = Ti.shape
            xi = X[:, i]
            xj = X[:, j]

            # validate 0-based and within target shape
            if xi.min() < 0 or xi.max() >= Ci:
                raise ValueError(
                    f"Column {i} has values outside [0,{Ci-1}] for constraint {(i,j)}."
                )
            if xj.min() < 0 or xj.max() >= Cj:
                raise ValueError(
                    f"Column {j} has values outside [0,{Cj-1}] for constraint {(i,j)}."
                )

            flat = xi * Cj + xj  # shape (N,)

            # Support: which (a,b) cells exist in df?
            support_counts = np.bincount(flat, minlength=Ci * Cj)
            support_mask = support_counts.reshape(Ci, Cj) > 0

            # If target puts mass where there is no support, infeasible
            missing_mass = float(Ti[~support_mask].sum())
            if missing_mass > 0:
                infeasible.append(
                    f"Constraint {(i,j)} infeasible: target assigns {missing_mass:.6g} "
                    f"probability to (a,b) pairs not present in df support."
                )

            precomp.append((i, j, Ci, Cj, Ti, flat, support_mask))

        # ---- Initialize weights ----
        w = np.full(N, 1.0 / N, dtype=np.float64)

        def compute_joint_from_flat(w_vec: np.ndarray, flat_idx: np.ndarray, Ci: int, Cj: int) -> np.ndarray:
            joint = np.bincount(flat_idx, weights=w_vec, minlength=Ci * Cj).reshape(Ci, Cj)
            return joint

        # ---- IPF iterations ----
        max_err_history = []
        for it in tqdm(range(max_iter)):
            # Apply one full sweep over constraints
            for (i, j, Ci, Cj, Ti, flat, support_mask) in precomp:
                J = compute_joint_from_flat(w, flat, Ci, Cj)

                # Avoid division by zero:
                # - Where support exists but J is ~0, add eps
                # - Where support does not exist, ratio shouldn't matter because no rows map there
                denom = np.where(support_mask, J, 1.0)
                denom = np.maximum(denom, eps)

                ratio = Ti / denom
                # For numerical stability, cap extreme ratios a bit (optional).
                # You can comment these two lines out if you prefer exact updates.
                ratio = np.clip(ratio, 0.0, 1.0 / eps)

                # Map each row to its cell ratio
                row_ratio = ratio.reshape(-1)[flat]  # shape (N,)

                if damping != 1.0:
                    # damped multiplicative update: w *= row_ratio^damping
                    w *= np.power(row_ratio, damping)
                else:
                    w *= row_ratio

                # Renormalize to keep weights in a sane scale
                s = w.sum()
                if s <= 0 or not np.isfinite(s):
                    raise FloatingPointError(
                        f"Weight normalization failed at iter {it} for constraint {(i,j)}."
                    )
                w /= s

            # Compute max error after sweep
            max_abs_err = 0.0
            mean_abs_err = 0.0
            n_constraints = 0

            for (i, j, Ci, Cj, Ti, flat, support_mask) in precomp:
                J = compute_joint_from_flat(w, flat, Ci, Cj)
                # Compare only on supported cells (unsupported cells can't be matched anyway)
                diff = np.abs((J - Ti)[support_mask])
                if diff.size > 0:
                    max_abs_err = max(max_abs_err, float(diff.max()))
                    mean_abs_err += float(diff.mean())
                    n_constraints += 1

            mean_abs_err = mean_abs_err / max(n_constraints, 1)
            max_err_history.append(max_abs_err)

            if verbose and (it % 25 == 0 or it == max_iter - 1):
                print(f"[iter {it:4d}] max_abs_err={max_abs_err:.3e}, mean_abs_err={mean_abs_err:.3e}")

            if max_abs_err < tol:
                break

        if normalize_weights:
            w /= w.sum()

        if not return_diagnostics:
            return w

        # ---- Diagnostics ----
        fitted = {}
        errors = {}
        for (i, j, Ci, Cj, Ti, flat, support_mask) in precomp:
            J = compute_joint_from_flat(w, flat, Ci, Cj)
            fitted[(i, j)] = J
            # store summary errors
            diff = (J - Ti)
            diff_supported = diff[support_mask]
            errors[(i, j)] = {
                "max_abs_err_supported": float(np.max(np.abs(diff_supported))) if diff_supported.size else 0.0,
                "mean_abs_err_supported": float(np.mean(np.abs(diff_supported))) if diff_supported.size else 0.0,
                "missing_target_mass_unsupported": float(Ti[~support_mask].sum()),
            }

        diagnostics = {
            "converged": (len(max_err_history) > 0 and max_err_history[-1] < tol),
            "iterations_used": len(max_err_history),
            "max_abs_err_history": max_err_history,
            "constraint_errors": errors,
            "infeasibility_warnings": infeasible,
            "fitted_joints": fitted,  # matrices
        }

        return w, diagnostics

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

class D4P_dataset(HouseholdPulse_dataset):
    def dummy(self):
        pass
    
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
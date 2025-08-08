import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
import itertools

#editable variables
GT_SIZE = 10000 #ground truth = GT
BIAS_SIZE = 1000
TARGET_VAR_GAP = 0.2 # >= |average(bias) - average(ground_truth)|
GT_SAVE_PATH = "./data/"
BIAS_SAVE_PATH = "./data/"

def random_sample_from_df(df,size):
    '''
    returns a new pandas data frame with "size" data points

    assumes target variable is the last column
    '''
    indexes = np.random.randint(low=0,high=df.shape[0],size=size)
    df1 = df.iloc[indexes,:]
    return df1.iloc[:,:-1], df1.iloc[:,-1].mean()

def force_sample_from_df(df,GT_size, BIAS_size,target_var_diff):
    '''
    IMPORTANT Assumes target variable in last column

    Will attempt to force a certain mean of target variable between GT and bias. 

    input:
        df - pandas dataframe
        GT_size - int
        BIAS_size - int
        target_var_diff - float, % difference between gt and bias avg target variables
    '''
    #gather target variable min and max
    target_min = df.iloc[:,-1].min()
    target_max = df.iloc[:,-1].max()
    target_diff_value = (target_max - target_min) * target_var_diff
    #start with initial sample
    indexes = np.random.randint(low=0,high=df.shape[0],size=BIAS_size)
    cur_bias = copy.deepcopy(df.iloc[indexes,:])
    indexes = np.random.randint(low=0,high=df.shape[0],size=GT_size)
    cur_gt = copy.deepcopy(df.iloc[indexes,:])

    cur_bias.sort_values(by=cur_bias.columns[-1],ascending=True,inplace=True)
    cur_gt.sort_values(by=cur_gt.columns[-1],ascending=True,inplace=True)

    cur_diff = np.abs(cur_bias.iloc[:,-1].mean() - cur_gt.iloc[:,-1].mean())
    while cur_diff < target_diff_value:
        indexes = np.random.randint(low=0,high=df.shape[0],size=BIAS_size)
        new_bias = copy.deepcopy(df.iloc[indexes,:])
        indexes = np.random.randint(low=0,high=df.shape[0],size=GT_size)
        new_gt = copy.deepcopy(df.iloc[indexes,:])
        new_bias.sort_values(by=cur_bias.columns[-1],ascending=True,inplace=True)
        new_gt.sort_values(by=cur_gt.columns[-1],ascending=True,inplace=True)

        bias_random_replacement = np.random.randint(0,BIAS_size//5)
        gt_random_replacement = np.random.randint(0,GT_size//5)

        #for bias replace the higher half with a random amount from the lower half (decrease avg)
        bias_random_indexes_replacement = np.random.randint(cur_bias.shape[0]//2,cur_bias.shape[0],
                                                            size=bias_random_replacement)
        new_bias_indexes = np.random.randint(0,cur_bias.shape[0]//2,size=bias_random_replacement)
        cur_bias.iloc[bias_random_indexes_replacement,:] = new_bias.iloc[new_bias_indexes,:]
        #for gt random replace the lower half with a random amount from the higher half (increase avg)
        gt_random_indexes_replacement = np.random.randint(0,cur_gt.shape[0]//2,size=gt_random_replacement)
        new_gt_indexes = np.random.randint(cur_gt.shape[0]//2,cur_gt.shape[0],size=gt_random_replacement)
        cur_gt.iloc[gt_random_indexes_replacement,:] = new_gt.iloc[new_gt_indexes,:]

        cur_bias.sort_values(by=cur_bias.columns[-1],ascending=True,inplace=True)
        cur_gt.sort_values(by=cur_gt.columns[-1],ascending=True,inplace=True)

        cur_diff = np.abs(cur_bias.iloc[:,-1].mean() - cur_gt.iloc[:,-1].mean())

        del new_bias
        del new_gt
    print(cur_diff)
    return cur_gt, cur_bias, cur_gt.iloc[:,-1].mean(), cur_bias.iloc[:,-1].mean()


def normalize_df(df):
    to_del = []
    for i in range(df.shape[1]):
        try:
            df.iloc[:,i].astype(float)
        except:
            assert i != df.shape[1]-1, "Error, last column is not float"
            to_del.append(i)
    df.drop(df.columns[to_del],axis=1,inplace=True)

    df_normalized = (df - df.min()) / (df.max() - df.min())
    columns_with_NaN = list(df_normalized.isna().any())
    return df_normalized, columns_with_NaN

def clean_NaN_by_col_index(df, NaN_list):
    '''
    drops the column in place for each df
    NaN_list - list[bool] - element k is true if column column k in df has NaN
    '''
    NaN_column_indexes = []
    for i in range(len(NaN_list)):
        if NaN_list[i] is True:
            NaN_column_indexes.append(i)
    if len(NaN_column_indexes) > 0:
        df.drop(df.columns[NaN_column_indexes],axis=1,inplace=True)

def clean_NaN_by_col_name(df, NaN_list):
    '''
    drops the column in place for each df
    NaN_list - list[bool] - element k is true if column column k in df has NaN
    '''
    df.drop(df[NaN_list],axis=1,inplace=True)


def clean_NaN_target_col(df):
    '''
    drops any rows with NaN in the last column (assumed target variable)
    '''
    df = df[df[df.columns[-1]].notna()]
    return df

def XBOX_get_GT_and_bias_ratios():
    #GT statistics
    #1 = male, 2 = female
    GT_sex = {1:0.48, 2:0.52}
    #age group index, 1:(18,29), 2:(30,44), 3:(45,64), 4:(65+)
    GT_age = {1:0.2,2:0.3,3:0.4,4:0.1}
    #race mapping 1:white, 2:black, everything else
    GT_race = {1:0.75, 2:0.1, 3:0.15}
    #< HS: 0-5, HS: 6, some_college: 7-9, finished college, 10-11
    GT_education = {1: 0.05, 2:0.15, 3: 0.3, 4: 0.5}

    global_GT_var_order = ['SEX', 'AGE', 'RACE', 'EDUC']
    global_GT_dict = []
    global_GT_dict.append(GT_sex)
    global_GT_dict.append(GT_age)
    global_GT_dict.append(GT_race)
    global_GT_dict.append(GT_education)

    #biased statistics
    global_bias_var_order = ['SEX', 'AGE', 'RACE', 'EDUC']
    bias_sex = {1:0.9, 2:0.1} #1 = male, 2 = female
    bias_age = {1:0.6,2:0.25,3:0.1,4:0.05}
    bias_race = {1:0.75, 2:0.1, 3:0.15}
    bias_education = {1: 0.05, 2:0.25, 3: 0.45, 4: 0.25}

    global_bias_dict = []
    global_bias_dict.append(bias_sex)
    global_bias_dict.append(bias_age)
    global_bias_dict.append(bias_race)
    global_bias_dict.append(bias_education)

    return global_GT_dict, global_GT_var_order,global_bias_dict,global_bias_var_order  

def get_all_persons_types_count(size,global_dict):
    '''
    takes a cartesian product of all keys in global_dict


    '''
    all_keys = [list(d.keys()) for d in global_dict]

    # Compute the Cartesian product of keys from all dictionaries
    cartesian_keys = list(itertools.product(*all_keys))

    # Create a new dictionary with the Cartesian product keys
    all_person_types_count = {key: None for key in cartesian_keys}

    for product_key in all_person_types_count:
        indpendent_joint_prob = 1
        for var_i, var_val in enumerate(product_key):
            indpendent_joint_prob *= global_dict[var_i][var_val]
        all_person_types_count[product_key] = indpendent_joint_prob*size

    all_person_types_count = {k:round(x) for k, x in all_person_types_count.items()}
    return all_person_types_count

def XBOX_get_sampled_df(var_order,all_person_types_count, df):
    #filter df by each specific type
    all_persons_dfs = []
    for person_type in all_person_types_count:
        condition = True
        for var_index, var_value in enumerate(person_type):
            condition = (condition & (df[var_order[var_index]] == var_value))
        
        conditional_df = df[condition]
        num_samples_needed = all_person_types_count[person_type]
        all_persons_dfs.append(conditional_df.sample(n=num_samples_needed))
    sampled_df = pd.concat(all_persons_dfs)
    return sampled_df

if __name__ == "__main__":
    #load and immediately purge any columns with NaN entries
    print("Loading and cleaning original CSV file...",end='')
    df = pd.read_csv("./data/cleanedCensusData.csv")
    df = clean_NaN_target_col(df)
    #columns_with_NaN = list(df.isna().any())
    #clean_NaN(df, columns_with_NaN)
    print("Done")

    #keep generating bias and gt data set until TARGET_VAR_GAP is reached
    print("Attempting to find target var gap of " + str(TARGET_VAR_GAP) + " or more...", end='')
    bias_df, gt_df = force_sample_from_df(df,GT_SIZE, BIAS_SIZE,TARGET_VAR_GAP)

    print("Done")
    print("Found datasets")

    #normalize bias and gt data set 
    bias_df_normalized, bias_NaN_columns = normalize_df(bias_df)
    gt_df_normalized, gt_df_NaN_columns = normalize_df(gt_df)
    all_NaNs_columns = bias_NaN_columns and gt_df_NaN_columns

    #purge any columns with NaN that are introduced as a result of normalization
    clean_NaN(bias_df_normalized, all_NaNs_columns)
    clean_NaN(gt_df_normalized, all_NaNs_columns)

    print(bias_df_normalized.shape, gt_df_normalized.shape)
    assert (bias_df_normalized.shape == gt_df_normalized.shape), "Print: bias and normalized shapes do not match"

    #save to CSV files
    bias_df_normalized.to_csv(BIAS_SAVE_PATH + "TargetVar="+str(bias_df_target_mean)+
                              ",NumPoints:"+str(BIAS_SIZE)+".csv",index=False)
    gt_df_normalized.to_csv(GT_SAVE_PATH + "TargetVar="+str(gt_df_target_mean)+
                              ",NumPoints:"+str(GT_SIZE)+".csv",index=False)


import pandas as pd
import numpy as np
import copy

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
    print(target_diff_value)
    #start with initial sample
    indexes = np.random.randint(low=0,high=df.shape[0],size=BIAS_size)
    cur_bias = copy.deepcopy(df.iloc[indexes,:])
    indexes = np.random.randint(low=0,high=df.shape[0],size=GT_size)
    cur_gt = copy.deepcopy(df.iloc[indexes,:])

    cur_bias.sort_values(by=cur_bias.columns[-1],ascending=True,inplace=True)
    cur_gt.sort_values(by=cur_gt.columns[-1],ascending=True,inplace=True)

    cur_diff = np.abs(cur_bias.iloc[:,-1].mean() - cur_gt.iloc[:,-1].mean())
    while cur_diff < target_diff_value:
        print(cur_diff)
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
    df_normalized = (df - df.min()) / (df.max() - df.min())
    columns_with_NaN = list(df_normalized.isna().any())
    return df_normalized, columns_with_NaN

def clean_NaN(df, NaN_list):
    '''
    drops the column in place for each df
    NaN_list - list[bool] - element k is true if column column k in df has NaN
    '''
    NaN_column_indexes = []
    for i in range(len(NaN_list)):
        if NaN_list[i] is True:
            NaN_column_indexes.append(i)
    df.drop(df.columns[NaN_column_indexes],axis=1,inplace=True)

if __name__ == "__main__":
    #load and immediately purge any columns with NaN entries
    print("Loading and cleaning original CSV file...",end='')
    df = pd.read_csv("./data/cleanedCensusData.csv")
    columns_with_NaN = list(df.isna().any())
    clean_NaN(df, columns_with_NaN)
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


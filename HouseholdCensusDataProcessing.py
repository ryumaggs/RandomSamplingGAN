import pandas as pd
import os
import random
from util import rename_check

'''
Important notes:
Previously processed data sets used, but this has been removed 12.10.25:
    #remove all people that got an associates degree. this info is not tracked by ipums
    survey_df= survey_df[survey_df['EEDUC'] != 5] #should i do this? idk

Survey todo: Age, Education, Income, marital status, race, region, SEX, RECVDVACC
Census todo: Age, Education, Income, marital status, race, region, sex
'''
def filter_questions(input_file, output_file):
    '''
    input_file - str - path to axios-ipsos covid survey
    output_file - str - path to output file

    it simply keeps the columns named in "columns_to_keep" and deletes all other columns
    '''
    print(input_file)
    df = pd.read_csv(input_file, encoding="ISO-8859-1")
    
    columns_to_keep = [
    "wt_final", "Q107_1", "Q107_2", "Q107_3", "Q107_4", "ppreg4", "ppmsacat", 
    "ppeducat","ppinc7", "ppgender", "ppmarit5", 
    "pphhsize", "pprent", "ppethm"
    ]
    #"ppinc7"
    # Filter the DataFrame to keep only the specified columns
    df_filtered = df[columns_to_keep]
    # Save the cleaned data
    df_filtered.to_csv(output_file, index=False, encoding="ISO-8859-1")
    print(f"Filtered data saved to {output_file}")

def ensure_zero_based_categories(df, col):
    """
    Ensures that a numeric column in a DataFrame representing categories
    has values starting at 0. If not, recodes so that the smallest value becomes 0.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    col : str
        The column name to check and fix.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the updated column.
    """
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric to represent categories.")

    min_val = df[col].min()
    if min_val > 0:
        df[col] = df[col] - min_val
        print(f"Column '{col}' recoded to start at 0.")
    else:
        print(f"Column '{col}' already starts at 0.")
    
    return df

def HHP_recode_birthyear_general(df, year, var_name='BIRTHYR', new_var_name="AGE"):
    def recode_census_age(value):
        if year - value >= 85: 
            return 4
        elif year - value >= 65:
            return 3
        elif year - value >= 50:
            return 2
        elif year - value >= 35:
            return 1
        elif year - value >= 18:
            return 0
        else:
            return None
    df[var_name] = df[var_name].apply(recode_census_age)
    rename_check(df, var_name, new_var_name)

def HHP_recode_education_survey(df, var_name='EEDUC', new_var_name='EDUC'):
    education_mapping = {
            1: 0,  # Less than high school
            2: 0,  # Less than high school
            3: 1,  # High school graduate
            4: 2,  # Some college
            6: 3,  # Bachelor’s degree
            7: 4   # Graduate degree
        }
    # Apply mapping
    df[var_name] = df[var_name].map(education_mapping)
    rename_check(df, var_name, new_var_name)

def HHP_recode_education_census(df, var_name='EDUC', new_var_name=None):
    
    educ_mapping = {
            0: 0,  # N/A or no schooling
            1: 0,  # Less than high school
            2: 0,  # Less than high school
            3: 0,  # Less than high school
            4: 0,  # Less than high school
            5: 0,  # Less than high school
            6: 1,  # High school graduate
            7: 2,  # Some college
            8: 2,  # Some college
            9: 2,  # Some college
            10: 3, # Bachelor's degree
            11: 4,  # Graduate degree,
            99: None,
    }
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def HHP_recode_income_survey(df, var_name='INCOME', new_var_name='INCTOT'):
    '''
    1) Less than $25,000  
    2) $25,000 - $34,999  
    3) $35,000 - $49,999   
    4) $50,000 - $74,999   
    5) $75,000 - $99,999   
    6) $100,000 - $149,999   
    7) $150,000 - $199,999
    8) $200,000 and above
    -99) Question seen but category not selected
    -88) Missing / Did not report
    '''
    
    def recode_income(value):
        # value is 1..24 corresponding to the fine-grained bins
        if value >= 1 and value <= 8:
            return value - 1
        else:
            return None  # -99 and -99 indicate missing/no answer
    df[var_name] = df[var_name].apply(recode_income)
    rename_check(df, var_name, new_var_name)
            
def HHP_recode_income_census(df, var_name='INCTOT', new_var_name=None):
    def map_income(value):
        if value == 9999999 or value == 9999998:
            return None
        if value < 25000:
            return 0  # Less than $25,000
        elif 25000 <= value < 35000:
            return 1  # $25,000 - $34,999
        elif 35000 <= value < 50000:
            return 2  # $35,000 - $49,999
        elif 50000 <= value < 75000:
            return 3  # $50,000 - $74,999
        elif 75000 <= value < 100000:
            return 4  # $75,000 - $99,999
        elif 100000 <= value < 150000:
            return 5  # $100,000 - $149,999
        elif 150000 <= value < 200000:
            return 6  # $150,000 - $199,999
        else:
            return 7  # $200,000 and above

    df[var_name] = df[var_name].apply(map_income)
    rename_check(df, var_name, new_var_name)

def HHP_recode_maritalStatus_survey(df, var_name='MS', new_var_name='MARST'):
    '''
    1) Now married 
    2) Widowed
    3) Divorced
    4) Separated
    5) Never married
    -99) Question seen but category not selected
    -88) Missing / Did not report
    '''

    marital_status_map = {
    1: 0,
    2: 1, 
    3: 2, 
    4: 3, 
    5: 4,
    -99: None,
    -88: None,
    }
    df[var_name] = df[var_name].map(marital_status_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_maritalStatus_census(df, var_name='MARST', new_var_name=None):
    marital_status_map = {
            1: 0,  # Married, spouse present -> Now married
            2: 0,  # Married, spouse absent -> Now married
            3: 3,  # Separated -> Separated
            4: 2,  # Divorced -> Divorced
            5: 1,  # Widowed -> Widowed
            6: 4,   # Never married/single -> Never married
            9: None,
    }
    df[var_name] = df[var_name].map(marital_status_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_perwt_census(df, var_name='PERWT', new_var_name=None):
    '''
    PERWT is a 6-digit numeric variable which indicates how many persons in the 
    U.S. population are represented by a given person in an IPUMS sample and has two implied decimals.
    '''
    def correct_perwt(value):
        return value/100
    df[var_name] = df[var_name].apply(correct_perwt)
    rename_check(df, var_name, new_var_name)

def HHP_recode_race_survey(df, var_name='RRACE', new_var_name='RACE'):
    '''
    1) White, Alone
    2) Black, Alone
    3) Asian, Alone
    4) Any other race alone, or race in combination
    '''
    raceA_to_unified = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    }
    df[var_name] = df[var_name].map(raceA_to_unified)
    rename_check(df, var_name, new_var_name)

def HHP_recode_race_census(df, var_name='RACE', new_var_name=None):
    #race census
    race_map = {
        1: 0,  # White -> White, Alone
        2: 1,  # Black/African American -> Black, Alone
        3: 3,  # American Indian or Alaska Native -> Any other race alone, or race in combination
        4: 2,  # Chinese -> Asian, Alone
        5: 2,  # Japanese -> Asian, Alone
        6: 2,  # Other Asian or Pacific Islander -> Asian, Alone
        7: 3,  # Other race, nec -> Any other race alone, or race in combination
        8: 3,  # Two major races -> Any other race alone, or race in combination
        9: 3   # Three or more major races -> Any other race alone, or race in combination
    }
    df[var_name] = df[var_name].map(race_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_region_survey(df, var_name='REGION', new_var_name=None):
    '''
    1) Northeast
    2) South
    3) Midwest
    4) West
    '''
    region_map = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    }
    df[var_name] = df[var_name].map(region_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_region_census(df, var_name='REGION', new_var_name=None):
    def recode_census_region(value):
        if 10 <= value <= 19:
            return 0  # NorthEast
        elif 20 <= value <= 29:
            return 2  # MidWest
        elif 30 <= value <= 39:
            return 1  # South
        elif 40 <= value <= 49:
            return 3  # West
        else:
            return None  # Handle unexpected values
    # Apply the recoding function to the region column in the census data
    df[var_name] = df[var_name].apply(recode_census_region)
    rename_check(df, var_name, new_var_name)

def HHP_recode_sex_survey(df, var_name=['EGENDER','EGENID_BIRTH'],new_var_name='SEX'):
    '''
    The variable name changed at some point during the survey waves
    1) Male 
    2) Female
    '''
    sex_map = {1:0, 2:1}
    for vname in var_name:
        if vname in df.columns:
            df[vname] = df[vname].map(sex_map)
            rename_check(df, vname, new_var_name)
            return

def HHP_recode_sex_census(df, var_name='SEX', new_var_name=None):
    sex_map = {1:0, 2:1, 9:None}
    df[var_name] = df[var_name].map(sex_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_vac_survey(df, var_name='RECVDVACC',new_var_name=None):
    '''
    1) Yes
    2) No
    -99) Question seen but category not selected
    -88) Missing / Did not report
    '''
    vac_map = {
        1: 1,
        2: 0,
        -99: None,
        -88: None,
    }
    df[var_name] = df[var_name].map(vac_map)
    rename_check(df, var_name, new_var_name)

def HHP_recode_survey(survey_df, year):
    HHP_recode_birthyear_general(survey_df,year,var_name='TBIRTH_YEAR')
    HHP_recode_education_survey(survey_df)
    HHP_recode_maritalStatus_survey(survey_df)
    HHP_recode_race_survey(survey_df)
    HHP_recode_region_survey(survey_df)
    HHP_recode_sex_survey(survey_df)
    HHP_recode_income_survey(survey_df)
    HHP_recode_vac_survey(survey_df)

def HHP_recode_census(census_df, year):
    #census specific fixes
    #HHP_recode_perwt_census(census_df) #scale perwt correctly (2 decimals implied)
    HHP_recode_birthyear_general(census_df,year,var_name='BIRTHYR')
    HHP_recode_education_census(census_df)
    HHP_recode_maritalStatus_census(census_df)
    HHP_recode_race_census(census_df)
    HHP_recode_region_census(census_df)
    HHP_recode_sex_census(census_df)
    HHP_recode_income_census(census_df)


'''
Survey age: TBIRTH_YEAR

'''
def recoding_survey_and_census_data(survey_df, census_df, target_var):

    '''
    survey_df - pandas df - household census survey data
    census_df - pandas df - IPUMS survey 
    iterates over a survey df and census df and re-encodes all variables manually
    target_var - list[str] of target variable names

    reincoding code was generated by CHATGPT
    '''
    state_code_to_biden_pct = {
        1: 0.3657,
        2: 0.4277,
        4: 0.4936,
        5: 0.3478,
        6: 0.6348,
        8: 0.5540,
        9: 0.5924,
        10: 0.5878,
        11: 0.9215,
        12: 0.4786,
        13: 0.4950,
        15: 0.6373,
        16: 0.3307,
        17: 0.5754,
        18: 0.4096,
        19: 0.4489,
        20: 0.4156,
        21: 0.3615,
        22: 0.3985,
        23: 0.5309,
        24: 0.6536,
        25: 0.6560,
        26: 0.5062,
        27: 0.5240,
        28: 0.4106,
        29: 0.4141,
        30: 0.4055,
        31: 0.3936,
        32: 0.5006,
        33: 0.5271,
        34: 0.5733,
        35: 0.5429,
        36: 0.6086,
        37: 0.4859,
        38: 0.3176,
        39: 0.4524,
        40: 0.3229,
        41: 0.5645,
        42: 0.5001,
        44: 0.5939,
        45: 0.4343,
        46: 0.3561,
        47: 0.3745,
        48: 0.4648,
        49: 0.3765,
        50: 0.6609,
        51: 0.5411,
        53: 0.5797,
        54: 0.2970,
        55: 0.4945,
        56: 0.2655
    }
    
    #drop all columns not prevelant to each survey
    YEAR = 2021
    relevant_columns_global = ['REGION', 'EDUC', 'INCTOT', 'SEX', 'MARST', 'RACE', 'AGE']
    #target_var = 'RECVDVACC' or HLTHINS1
    combined_census_df=None
    if survey_df is not None:
        HHP_recode_survey(survey_df, YEAR)
        survey_df = survey_df.filter(items=target_var + relevant_columns_global)
        survey_df = survey_df.dropna()
    if census_df is not None:
        HHP_recode_census(census_df, YEAR)
        #census_df['PERWT'] = census_df['PERWT']
        census_df = census_df[census_df['GQ'] == 1] #remove any institutionalized persons
        census_df = census_df.filter(items=['PERWT'] + relevant_columns_global)
        census_df = census_df.dropna()

        #aggregate by unique data points, sum PERWT's
        grouped_df = census_df.groupby(list(census_df.columns[1:]), as_index=False)['PERWT'].sum()
        # Reorder columns so PERWT is first
        cols = ['PERWT'] + [c for c in grouped_df.columns if c != 'PERWT']
        combined_census_df = grouped_df[cols]

    return survey_df, census_df, combined_census_df

def load_evenly_sampled_csv_rows(directory_path, K, target_var):
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    num_files = len(csv_files)
    
    if num_files == 0:
        raise ValueError("No CSV files found in the directory.")
    
    # Determine how many rows to sample from each file
    base_rows = K // num_files
    remainder = K % num_files  # Distribute leftover rows evenly
    sampled_rows = []

    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        
        if "EGENID_BIRTH" in df.columns:
            df = df.rename(columns={"EGENID_BIRTH": "EGENDER"})
            
        nan_check = [target_var, 'REGION', 'EEDUC', 'INCOME', 'EGENDER', 'MS', 'RRACE', 'TBIRTH_YEAR', ]
        df = df.dropna(subset=nan_check)
        df.drop(df[~df[target_var].isin([1, 2])].index, inplace=True)

        if df.empty:
            continue  # Skip empty files

        # Determine how many rows to sample from this file
        n_rows = base_rows + (1 if i < remainder else 0)
        n_rows = min(n_rows, len(df))  # Don't exceed available rows

        sampled_df = df.sample(n=n_rows, random_state=random.randint(0, 10000))
        sampled_rows.append(sampled_df)

    # Concatenate all sampled dataframes
    combined_df = pd.concat(sampled_rows, ignore_index=True)
    
    # If we sampled fewer rows than K due to small CSVs, fill in extra from any available data
    while len(combined_df) < K:
        for csv_file in csv_files:
            if len(combined_df) >= K:
                break
            file_path = os.path.join(directory_path, csv_file)
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            remaining_rows = df[~df.index.isin(sampled_df.index)]
            if not remaining_rows.empty:
                extra_row = remaining_rows.sample(n=1, random_state=random.randint(0, 10000))
                combined_df = pd.concat([combined_df, extra_row], ignore_index=True)

    return combined_df.head(K)

if __name__ == "__main__":
    pass
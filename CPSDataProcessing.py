import pandas as pd
import numpy as np
from util import rename_check

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


def CPS_recode_age_general(df, var_name='AGE', new_var_name=None):
    def recode_census_age(value):
        if value >= 85: 
            return 4
        elif value >= 65:
            return 3
        elif value >= 50:
            return 2
        elif value >= 35:
            return 1
        elif value >= 18:
            return 0
        else:
            return None
    df[var_name] = df[var_name].apply(recode_census_age)
    rename_check(df, var_name, new_var_name)

def CPS_recode_birthyear_general(df, year, var_name='TBIRTH_YEAR', new_var_name="AGE"):
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
    df[var_name] = df[var_name].apply(recode_census_age)
    rename_check(df, var_name, new_var_name)

def CPS_recode_education_survey(df, var_name='EEDUC', new_var_name='EDUC'):
    educ_collapse_mapping = {
    1: 0,  # Less than high school → 0
    2: 0,  # Some high school → 0
    3: 1,  # High school graduate → 1
    4: 2,  # Some college, no degree → 2
    5: 2,  # Associate’s degree → 2
    6: 3,  # Bachelor’s → 3
    7: 4,   # Graduate → 4
    -99: None,
    -88: None,
    }
    # Apply mapping
    df[var_name] = df[var_name].map(educ_collapse_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_education_census(df, var_name='EDUC', new_var_name=None):
    educ_mapping = {
        # Less than high school
        0: 0, 1: 0, 2: 0,
        10: 0, 11: 0, 12: 0, 13: 0, 14: 0,    # Grades 1–4
        20: 0, 21: 0, 22: 0,                  # Grades 5–6
        30: 0, 31: 0, 32: 0,                  # Grades 7–8
        40: 0, 50: 0, 60: 0, 70: 0, 71: 0, 72: 0,  # Grades 9–12 (no diploma)

        # High school graduate or equivalent
        73: 1,

        # Some college (no bachelor’s)
        80: 2, 81: 2,
        90: 2, 91: 2, 92: 2,
        100: 2,

        # Bachelor's degree
        110: 3, 111: 3,

        # Graduate degree (master’s, professional, doctorate)
        120: 4, 121: 4, 122: 4, 123: 4, 124: 4, 125: 4,

        # Missing
        999: None
    }
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_employment_survey(df, var_name='ANYWORK', new_var_name='EMPSTAT'):
    def recode_anywork(value):
        if value == 2: #no
            return 0
        elif value == 1: #yes
            return 1
        else:
            return None
    df['ANYWORK'] = df['ANYWORK'].apply(recode_anywork)
    rename_check(df, var_name, new_var_name)

def CPS_recode_employment_census(df, var_name='EMPSTAT', new_var_name=None):
    def encode_employment(value):
            if value == 10:
                return 1
            else:
                return 0
            
    df['EMPSTAT'] = df['EMPSTAT'].apply(encode_employment)
    rename_check(df, var_name, new_var_name)
    
def CPS_recode_maritalStatus_survey(df, var_name='MS', new_var_name='MARST'):
    marst_mapping = {
    1: 0,  # Married, spouse present → Now married (was 1)
    2: 2,  # Married, spouse absent → Now married (was 1)
    3: 2,  # Separated → Separated (was 4)
    4: 3,  # Divorced → Divorced (was 3)
    5: 4,  # Widowed → Widowed (was 2)
    -99: None,  # NIU → Missing
    -88: None, #never answered
    }
    df[var_name] = df[var_name].map(marst_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_maritalStatus_census(df, var_name='MARST', new_var_name=None):
    marst_mapping = {
    1: 0,  # Married, spouse present → Now married (was 1)
    2: 0,  # Married, spouse absent → Now married (was 1)
    3: 3,  # Separated → Separated (was 4)
    4: 2,  # Divorced → Divorced (was 3)
    5: 1,  # Widowed → Widowed (was 2)
    6: 4,  # Never married/single → Never married (was 5)
    7: 1,  # Widowed or Divorced → Widowed (was 2)
    9: None  # NIU → Missing
    }
    df[var_name] = df[var_name].map(marst_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_race_survey(df, var_name='RRACE', new_var_name='RACE'):
    df[var_name] = df[var_name] - 1
    rename_check(df, var_name, new_var_name)

def CPS_recode_race_census(df, var_name='RACE', new_var_name=None):
    race_mapping = {
    100: 0,  # White alone
    200: 1,  # Black alone
    650: 2,  # Asian or Pacific Islander (old combined category)
    651: 2,  # Asian only

    # Everything else
    300: 3,  # American Indian / Aleut / Eskimo
    652: 3,  # Hawaiian or Pacific Islander only
    700: 3,  # Other (single) race
    801: 3, 802: 3, 803: 3, 804: 3, 805: 3, 806: 3, 807: 3,
    808: 3, 809: 3, 810: 3, 811: 3, 812: 3, 813: 3, 814: 3,
    815: 3, 816: 3, 817: 3, 818: 3, 819: 3,
    820: 3, 830: 3,
    999: None  # Missing / Blank
    }
    df[var_name] = df[var_name].map(race_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_region_survey(df, var_name='REGION', new_var_name=None):
    region_labels_0based = {
    1: 0, #"Northeast",
    2: 1, #"South",
    3: 2, #"Midwest",
    4: 3, #"West"
    }
    df[var_name] = df[var_name].map(region_labels_0based)
    rename_check(df, var_name, new_var_name)

def CPS_recode_region_census(df, var_name='REGION', new_var_name=None):
    region_mapping = {
    11: 0,  # New England → Northeast
    12: 0,  # Middle Atlantic → Northeast
    21: 2,  # East North Central → Midwest
    22: 2,  # West North Central → Midwest
    31: 1,  # South Atlantic → South
    32: 1,  # East South Central → South
    33: 1,  # West South Central → South
    41: 3,  # Mountain → West
    42: 3,  # Pacific → West
    97: None  # State not identified → Missing
    }
    df["REGION"] = df["REGION"].map(region_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_sex_general(df, var_name='SEX', new_var_name=None):
    sex_mapping = {
    1: 0, #male
    2: 1, #female
    }
    df[var_name] = df[var_name].map(sex_mapping)
    rename_check(df, var_name, new_var_name)

def CPS_recode_survey(survey_df, year):
    CPS_recode_birthyear_general(survey_df, year)
    CPS_recode_education_survey(survey_df)
    CPS_recode_maritalStatus_survey(survey_df)
    CPS_recode_race_survey(survey_df)
    CPS_recode_region_survey(survey_df)
    CPS_recode_sex_general(survey_df, var_name='EGENID_BIRTH', new_var_name='SEX')
    CPS_recode_employment_survey(survey_df)

def CPS_recode_census(census_df, year):
    CPS_recode_age_general(census_df)
    CPS_recode_education_census(census_df)
    CPS_recode_maritalStatus_census(census_df)
    CPS_recode_race_census(census_df)
    CPS_recode_region_census(census_df)
    CPS_recode_sex_general(census_df, var_name='SEX')
    CPS_recode_employment_census(census_df)
    
def CPS_recoding_survey_and_census_data(survey_df, census_df, target_var):

    '''
    survey_df - pandas df - household census survey data
    census_df - pandas df - IPUMS survey 
    iterates over a survey df and census df and re-encodes all variables manually

    reincoding code was generated by CHATGPT
    '''

    #drop all columns not prevelant to each survey
    year = 2024
    relevant_columns = ['REGION', 'AGE', 'SEX', 'RACE', 'EDUC', 'MARST']
    if survey_df is not None:
        CPS_recode_survey(survey_df, year)
        survey_df = survey_df.filter(items=target_var + relevant_columns)
        survey_df = survey_df.dropna()

    if census_df is not None:
        CPS_recode_census(census_df, year)
        census_df.rename(columns={'WTFINL':'PERWT'},inplace=True)
        census_df['PERWT'] = census_df['PERWT']
        census_df = census_df.filter(items=['PERWT'] + relevant_columns)
        census_df = census_df.dropna()

    return survey_df, census_df

if __name__ == "__main__":
    pass
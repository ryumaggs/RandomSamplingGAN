import pandas as pd
from util import rename_check

def axios_recode_age_general(df, var_name='ppage', new_var_name="AGE"):
    def recode_census_age(v):
        try:
            value = int(v)
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
                return None #cut out 17 and younger responses
        except (ValueError, TypeError):
            return None

    df[var_name] = df[var_name].apply(recode_census_age)
    rename_check(df, var_name, new_var_name)

def axios_recode_birthyear_census(df, year, var_name='BIRTHYR', new_var_name="AGE"):
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

def axios_recode_education_survey(df, var_name='ppeducat', new_var_name='EDUC'):
    ppeducat_mapping = {
    'Bachelors degree or higher': 3,
    'Some college': 2,
    'High school': 1,
    'Less than high school': 0
    }

    # Recode the ppeducat column in the survey data
    # Apply mapping
    df[var_name] = df[var_name].map(ppeducat_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_education_census(df, var_name='EDUC', new_var_name=None):
    educ_mapping = {
    0: 0,  # N/A or no schooling → Less than high school
    1: 0,  # Nursery school to grade 4 → Less than high school
    2: 0,  # Grade 5, 6, 7, or 8 → Less than high school
    3: 0,  # Grade 9 → Less than high school
    4: 0,  # Grade 10 → Less than high school
    5: 0,  # Grade 11 → Less than high school
    6: 1,  # Grade 12 → High school
    7: 2,  # 1 year of college → Some college
    8: 2,  # 2 years of college → Some college
    9: 2,  # 3 years of college → Some college
    10: 3,  # 4 years of college → Bachelors degree or higher
    11: 3   # 5+ years of college → Bachelors degree or higher
    }

    # Recode the EDUC column in the census data
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_famsize_survey(df, var_name='pphhsize', new_var_name='FAMSIZE'):
    #family size survey
    pphhsize_mapping = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6 or more': 5
    }

    # Recode the pphhsize column in the survey data
    df[var_name] = df[var_name].map(pphhsize_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_famsize_census(df, var_name='FAMSIZE', new_var_name=None):
    def recode_famsize(size):
        if size >= 6:
            return 5  # '6 or more'
        else:
            return size-1  # Keep values 1-5 as is

    # Apply the recoding function to the FAMSIZE column
    df[var_name] = df[var_name].apply(recode_famsize)
    rename_check(df, var_name, new_var_name)
                                   
def axios_recode_income_survey(df, var_name='ppinc7', new_var_name='INCTOT'):
    ppinc7_mapping = {
        '$150,000 or more' : 6,
        '$100,000 to $149,999' : 5,
        '$75,000 to $99,999' : 4,
        '$50,000 to $74,999' : 3,
        '$25,000 to $49,999' : 2,
        '$10,000 to $24,999' : 1,
        'Less than $10,000' : 0,

        'Less than $5,000': 0,
        '$5,000 to $7,499': 0,
        '$7,500 to $9,999': 0,
        '$10,000 to $12,499': 1,
        '$12,500 to $14,999': 1,
        '$15,000 to $19,999': 1,
        '$20,000 to $24,999': 1,
        '$25,000 to $29,999': 2,
        '$30,000 to $34,999': 2,
        '$35,000 to $39,999': 2,
        '$40,000 to $49,999': 2,
        '$50,000 to $59,999': 3,
        '$60,000 to $74,999': 3,
        '$75,000 to $84,999': 4,
        '$85,000 to $99,999': 4,
        '$100,000 to $124,999': 5,
        '$125,000 to $149,999': 5,
        '$150,000 to $174,999': 6,
        '$175,000 to $199,999': 6,
        '$200,000 to $249,999': 6,
        '$250,000 or more': 6
    }
    
    df[var_name] = df[var_name].map(ppinc7_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_income_census(df, var_name='INCTOT', new_var_name=None):
    def map_income_to_code(income):
        if income < 10000:
            return 0
        elif 10000 <= income <= 24999:
            return 1
        elif 25000 <= income <= 49999:
            return 4
        elif 50000 <= income <= 74999:
            return 3
        elif 75000 <= income <= 99999:
            return 2
        elif 100000 <= income <= 149999:
            return 5
        elif income >= 150000:
            return 6
        else:
            return None  # in case income is negative or invalid
    df[var_name] = df[var_name].apply(map_income_to_code)
    rename_check(df, var_name, new_var_name)

def axios_recode_maritalStatus_survey(df, var_name='ppmarit5', new_var_name='MARST'):
    '''
    Married                626
    Never married          193
    Divorced               101
    Widowed                 55
    Living with partner     47
    Separated               16
    '''
    ppmarit5_mapping = {
        'Now Married': 0,
        'Married': 0, 
        'Separated': 1,
        'Divorced': 2,
        'Widowed': 3,
        'Never married': 4,
        'Living with partner': 0
        }

    # Recode the ppmarit5 column in the survey data
    
    df[var_name] = df[var_name].map(ppmarit5_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_maritalStatus_census(df, var_name='MARST', new_var_name=None):
    #survey marital
    
    marst_mapping = {
    1: 0,  # Married, spouse present → Married
    2: 0,  # Married, spouse absent → Married
    3: 1,  # Separated → Separated
    4: 2,  # Divorced → Divorced
    5: 3,  # Widowed → Widowed
    6: 4   # Never married/single → Never married/single
    }

    # Recode the MARST column in the census data
    df[var_name] = df[var_name].map(marst_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_metro_survey(df, var_name='ppmsacat', new_var_name='METRO'):

        # Define a mapping for the ppmsacat values
        ppmsacat_mapping = {
            'Metro': 1,      # In metropolitan area
            'Non-Metro': 0   # Not in metropolitan area
        }

        # Recode the ppmsacat column in the survey data
        df[var_name] = df[var_name].map(ppmsacat_mapping)
        rename_check(df, var_name, new_var_name)

def axios_recode_metro_census(df, var_name='METRO', new_var_name=None):
    # Define a function to recode METRO values
    def recode_metro(value):
        if value in [2, 3, 4]:
            return 1  # In metropolitan area
        elif value == 1:
            return 0  # Not in metropolitan area
        elif value == 0:
            return None  # Metropolitan status indeterminable (can be dropped or handled separately)
        else:
            return None  # Handle unexpected values

    # Apply the recoding function to the METRO column in the census data
    df[var_name] = df[var_name].apply(recode_metro)
    rename_check(df, var_name, new_var_name)

def axios_recode_race_survey(df, var_name='ppethm', new_var_name='RACE'):
    ppethm_mapping = {
        'White, Non-Hispanic': 0,
        'Black, Non-Hispanic': 1,
        'Hispanic': 2,
        'Other, Non-Hispanic': 3,
        '2+ Races, Non-Hispanic': 4
        }
    df[var_name] = df[var_name].map(ppethm_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_race_census(df, var_name='RACE', new_var_name=None):
    #race census
    race_mapping = {
    1: 0,  # White → 1
    2: 1,  # Black/African American → 2
    3: 2,  # American Indian or Alaska Native → 4 (Other, Non-Hispanic)
    4: 3,  # Chinese → 4 (Other, Non-Hispanic)
    5: 3,  # Japanese → 4 (Other, Non-Hispanic)
    6: 3,  # Other Asian or Pacific Islander → 4 (Other, Non-Hispanic)
    7: 3,  # Other race, nec → 4 (Other, Non-Hispanic)
    8: 4,  # Two major races → 5 (2+ Races, Non-Hispanic)
    9: 4   # Three or more major races → 5 (2+ Races, Non-Hispanic)
    }
    df[var_name] = df[var_name].map(race_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_region_survey(df, var_name='ppreg4', new_var_name="REGION"):
    region_mapping_survey = {
        'MidWest': 0,
        'South': 1,
        'West': 2,
        'NorthEast': 3
        }
    
    df[var_name] = df[var_name].map(region_mapping_survey)
    rename_check(df, var_name, new_var_name)

def axios_recode_region_census(df, var_name='REGION', new_var_name=None):
    def recode_census_region(value):
        if 10 <= value <= 19:
            return 3  # NorthEast
        elif 20 <= value <= 29:
            return 0  # MidWest
        elif 30 <= value <= 39:
            return 1  # South
        elif 40 <= value <= 49:
            return 2  # West
        else:
            return None  # Handle unexpected values
    # Apply the recoding function to the region column in the census data
    df[var_name] = df[var_name].apply(recode_census_region)
    rename_check(df, var_name, new_var_name)

def axios_recode_sex_survey(df, var_name='ppgender', new_var_name='SEX'):
    gender_mapping = {'Male': 0, 'Female':1}

    df[var_name] = df[var_name].map(gender_mapping)
    rename_check(df, var_name, new_var_name)
        
def axios_recode_sex_census(df, var_name='SEX', new_var_name=None):
    sex_mapping = {
    1: 0, #male
    2: 1, #female
    }
    df[var_name] = df[var_name].map(sex_mapping)
    rename_check(df, var_name, new_var_name)

def axios_recode_vac_survey(df, var_name='Q107_1', new_var_name='RECVDVACC'):
    vaccine_mapping = {
            'Yes, I have received the vaccine': 1,
            'Skipped': 0
    }
    df[var_name] = df[var_name].map(vaccine_mapping)
    rename_check(df, var_name, new_var_name)
    
def axios_recode_survey(survey_df, year):
    axios_recode_age_general(survey_df)
    axios_recode_education_survey(survey_df)
    if 'ppmarit5' in survey_df:
        axios_recode_maritalStatus_survey(survey_df)
    else:
        axios_recode_maritalStatus_survey(survey_df,var_name='ppmarit')
    axios_recode_race_survey(survey_df)
    axios_recode_region_survey(survey_df)
    axios_recode_sex_survey(survey_df)

    axios_recode_famsize_survey(survey_df)
    if 'ppinc7' in survey_df:
        axios_recode_income_survey(survey_df)
    else:
        axios_recode_income_survey(survey_df,var_name='ppincimp')
    axios_recode_vac_survey(survey_df)
    #axios_recode_metro_survey(survey_df)
    

def axios_recode_census(census_df, year):
    axios_recode_birthyear_census(census_df, year)
    axios_recode_education_census(census_df)
    axios_recode_maritalStatus_census(census_df)
    axios_recode_race_census(census_df)
    axios_recode_region_census(census_df)
    axios_recode_sex_census(census_df)

    axios_recode_famsize_census(census_df)
    axios_recode_income_census(census_df)
    #axios_recode_metro_census(census_df)

def axios_recoding_survey_and_census_data(survey_df, census_df, target_var):

    '''
    survey_df - pandas df - household census survey data
    census_df - pandas df - IPUMS survey 
    target_var - list[str] - list of target variables
    iterates over a survey df and census df and re-encodes all variables manually


    columns_to_keep = [
    "wt_final", "Q107_1", "Q107_2", "Q107_3", "Q107_4", "ppreg4", "ppmsacat", 
    "ppeducat","ppinc7", "ppgender", "ppmarit5", 
    "pphhsize", "pprent", "ppethm"
    ]

    '''

    #drop all columns not prevelant to each survey
    relevant_columns = ['AGE', 'REGION', 'EDUC', 'SEX', 'MARST', 'RACE', 'FAMSIZE', 'INCTOT']
    year = 2021
    if survey_df is not None:
        axios_recode_survey(survey_df, year)
        survey_df = survey_df.filter(items=target_var + relevant_columns)
        survey_df = survey_df.dropna()

    if census_df is not None:
        axios_recode_census(census_df, year)
        census_df.rename(columns={'WTFINL':'PERWT'},inplace=True)
        census_df['PERWT'] = census_df['PERWT']
        census_df = census_df.filter(items=['PERWT'] + relevant_columns)
        census_df = census_df.dropna()

        #aggregate by unique data points, sum PERWT's
        grouped_df = census_df.groupby(list(census_df.columns[1:]), as_index=False)['PERWT'].sum()
        # Reorder columns so PERWT is first
        cols = ['PERWT'] + [c for c in grouped_df.columns if c != 'PERWT']
        combined_census_df = grouped_df[cols]


    return survey_df, census_df, combined_census_df

def load_csv(path_to_survey, path_to_census):
    '''
    IMPORTANT. axios ipsos data must be loaded with the encoding "ISO-8859-1"
    '''
    df_survey = pd.read_csv(path_to_survey, encoding="ISO-8859-1")
    return df_survey
    

        

if __name__ == "__main__":
    pass
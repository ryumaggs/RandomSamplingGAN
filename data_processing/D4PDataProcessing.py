import pandas as pd
from util import rename_check

def d4p_recode_age_general(df, var_name='age', new_var_name="AGE"):
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

def d4p_recode_birthyear_census(df, year, var_name='BIRTHYR', new_var_name="AGE"):
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

def d4p_recode_education_survey(df, var_name='education', new_var_name='EDUC'):
    educ_mapping = {
        -3105: None,  # None / missing
        1: 0,      #less than high school
        2: 1,      # finished highscool
        3: 2,      # some amount of college
        4: 2,      # S
        5: 2,      # 
        6: 3,      # Bachelor's degree
        7: 4,      # Beyond college
        8: 4       # Beyond College
    }
    # Apply mapping
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def d4p_recode_education_census(df, var_name='EDUC', new_var_name=None):
    educ_mapping = {
        0: 0,    # 00 N/A or no schooling -> None/missing
        1: 0,    # 01 Nursery–Grade4 -> Primary
        2: 0,    # 02 Grade5–8 -> Primary
        3: 0,    # 03 Grade9 -> Some HS
        4: 0,    # 04 Grade10 -> Some HS
        5: 0,    # 05 Grade11 -> Some HS
        6: 1,    # 06 Grade12 -> HS graduate
        7: 2,    # 07 1 year college -> Some college/vocational
        8: 2,    # 08 2 years college -> Associate's
        9: 2,    # 09 3 years college -> Some college/vocational
        10: 3,   # 10 4 years college -> Bachelor's
        11: 4,   # 11 5+ years college -> Graduate degree
        99: None    # 99 Missing -> None/missing
    }
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def d4p_recode_income_survey(df, var_name='hhi', new_var_name='INCTOT'):
    def recode_income(value):
        # value is 1..24 corresponding to the fine-grained bins
        if value in [1, 2, 3]:
            return 0  # Less than $25,000
        elif value in [4, 5]:
            return 1  # $25,000 - $34,999
        elif value in [6, 7, 8]:
            return 2  # $35,000 - $49,999
        elif value in [9, 10, 11, 12, 13]:
            return 3  # $50,000 - $74,999
        elif value in [14, 15, 16, 17, 18]:
            return 4  # $75,000 - $99,999
        elif value in [19, 20]:
            return 5  # $100,000 - $149,999
        elif value in [21, 22]:
            return 6  # $150,000 - $199,999
        elif value in [23, 24]:
            return 7  # $200,000 and above
        else:
            return None  # For any unexpected value
    df[var_name] = df[var_name].apply(recode_income)
    rename_check(df, var_name, new_var_name)
            
def d4p_recode_income_census(df, var_name='INCTOT', new_var_name=None):
    def map_income(value):
            if value == -99:
                return -99  # Question seen but not selected
            elif value == -88:
                return -88  # Missing / Did not report
            elif value < 25000:
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

def d4p_recode_maritalStatus_survey(df, var_name='marstat', new_var_name='MARST'):
    marital_status_map = {
    "Married": 0,
    "Domestic /civil partnership": 0,
    "Never married": 4,
    "Divorced": 2,
    "Widowed": 3,
    "Separated": 1
    }
    df[var_name] = df[var_name].map(marital_status_map)
    rename_check(df, var_name, new_var_name)

def d4p_recode_maritalStatus_census(df, var_name='MARST', new_var_name=None):
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

def d4p_recode_race_survey(df, var_name='ethnicity', new_var_name='RACE'):
    raceA_to_unified = {
    1: 0,   # White
    2: 1,   # Black / African American
    3: 2,   # American Indian / Alaska Native
    4: 5,   # Asian Indian -> Other Asian/Pacific Islander
    5: 3,   # Chinese
    6: 5,   # Filipino -> Other Asian/Pacific Islander
    7: 4,   # Japanese
    8: 5,   # Korean -> Other Asian/Pacific Islander
    9: 5,   # Vietnamese -> Other Asian/Pacific Islander
    10: 5,  # Other Asian -> Other Asian/Pacific Islander
    11: 5,  # Native Hawaiian -> Other Asian/Pacific Islander
    12: 5,  # Guamanian -> Other Asian/Pacific Islander
    13: 5,  # Samoan -> Other Asian/Pacific Islander
    14: 5,  # Other Pacific Islander -> Other Asian/Pacific Islander
    15: 6   # Some other race -> Other race
    }
    

    df[var_name] = df[var_name].map(raceA_to_unified)
    rename_check(df, var_name, new_var_name)

def d4p_recode_race_census(df, var_name='RACE', new_var_name=None):
    race_mapping = {
        1: 0,   # White
        2: 1,   # Black / African American
        3: 2,   # American Indian / Alaska Native
        4: 3,   # Chinese
        5: 4,   # Japanese
        6: 5,   # Other Asian or Pacific Islander
        7: 6,   # Other race, nec
        8: 6,   # Two major races -> Multiple races
        9: 6    # Three or more major races -> Multiple races
    }
    df[var_name] = df[var_name].map(race_mapping)
    rename_check(df, var_name, new_var_name)

def d4p_recode_region_survey(df, var_name='region', new_var_name="REGION"):
    def get_region_id(state_abbr):
        region_mapping = {
            1: 0, #northeast
            2: 1, #Midwest
            3: 2, #south
            4: 3, #west
        }
        return region_mapping.get(state_abbr, None)
    df[var_name] = df[var_name].apply(get_region_id)
    rename_check(df, var_name, new_var_name)

def d4p_recode_region_census(df, var_name='REGION', new_var_name=None):
    region_mapping = {
    11: 0,  # New England → Northeast
    12: 0,  # Middle Atlantic → Northeast
    13: 0,
    21: 1,  # East North Central → Midwest
    22: 1,  # West North Central → Midwest
    23: 1, 
    31: 2,  # South Atlantic → South
    32: 2,  # East South Central → South
    33: 2,  # West South Central → South
    34: 2, 
    41: 3,  # Mountain → West
    42: 3,  # Pacific → West
    43: 3, 
    91: None, 
    92: None, 
    97: None,
    97: None  # State not identified → Missing
    }
    df["REGION"] = df["REGION"].map(region_mapping)
    rename_check(df, var_name, new_var_name)

def d4p_recode_sex_survey(df, var_name='gender', new_var_name='SEX'):
    def recode_sex(value):
        if value == 'Female':
            return 1
        elif value == 'Male':
            return 0
        else:
            return None
    df[var_name] = df[var_name].apply(recode_sex)
    rename_check(df, var_name, new_var_name)
            
def d4p_recode_sex_census(df, var_name='SEX', new_var_name=None):
    sex_mapping = {
    1: 0, #male
    2: 1, #female
    }
    df[var_name] = df[var_name].map(sex_mapping)
    rename_check(df, var_name, new_var_name)

def d4p_recode_vac_survey(df, var_name='vax', new_var_name='RECVDVACC'):
    
    def encode_employment(value):
        try:
            value = str(value)
            if 'no' in value.lower():
                return 0
            if 'yes' in value.lower():
                return 1
        except (ValueError, TypeError):
            return None
    
    df[var_name] = df[var_name].apply(encode_employment)
    rename_check(df, var_name, new_var_name)

def d4p_recode_survey(survey_df, year):
    d4p_recode_age_general(survey_df)
    d4p_recode_education_survey(survey_df)
    d4p_recode_maritalStatus_survey(survey_df)
    d4p_recode_race_survey(survey_df)
    d4p_recode_region_survey(survey_df)
    d4p_recode_sex_survey(survey_df)
    d4p_recode_income_survey(survey_df)
    d4p_recode_vac_survey(survey_df)

def d4p_recode_census(census_df, year):
    d4p_recode_birthyear_census(census_df,year,var_name='BIRTHYR')
    d4p_recode_education_census(census_df)
    d4p_recode_maritalStatus_census(census_df)
    d4p_recode_race_census(census_df)
    d4p_recode_region_census(census_df)
    d4p_recode_sex_census(census_df)
    d4p_recode_income_census(census_df)

def recoding_survey_and_census_data(survey_df, census_df, target_var):

    '''
    survey_df - pandas df - household census survey data
    census_df - pandas df - IPUMS survey 
    target_var - list[str] - list of target variables
    iterates over a survey df and census df and re-encodes all variables manually

    reincoding code was generated by CHATGPT
    '''
    #drop all columns not prevelant to each survey
    relevant_columns = ['AGE', 'REGION', 'EDUC', 'SEX', 'MARST', 'RACE', 'INCTOT']
    year = 2021
    if survey_df is not None:
        d4p_recode_survey(survey_df, year)
        survey_df = survey_df.filter(items=target_var + relevant_columns)
        survey_df = survey_df.dropna()

    if census_df is not None:
        d4p_recode_census(census_df, year)
        if 'WTFINL' in census_df.columns:
            census_df.rename(columns={'WTFINL':'PERWT'},inplace=True)
        census_df = census_df[census_df['GQ'] == 1]
        #census_df['PERWT'] = census_df['PERWT']
        census_df = census_df.filter(items=['PERWT'] + relevant_columns)
        census_df = census_df.dropna()
                
        grouped_df = census_df.groupby(list(census_df.columns[1:]), as_index=False)['PERWT'].sum()
        # Reorder columns so PERWT is first
        cols = ['PERWT'] + [c for c in grouped_df.columns if c != 'PERWT']
        combined_census_df = grouped_df[cols]

    return survey_df, census_df, combined_census_df
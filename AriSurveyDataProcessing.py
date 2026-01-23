import pandas as pd
from util import rename_check

def lucid_recode_age_general(df, var_name='age', new_var_name="AGE"):
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

def lucid_recode_birthyear_census(df, year, var_name='BIRTHYR', new_var_name="AGE"):
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

def lucid_recode_education_survey(df, var_name='education', new_var_name='EDUC'):
    educ_mapping = {
        'Less than high school': 0,  # N/A or no schooling
        'Some high school': 0,
        'High school graduate or equivalent (for example GED)': 1,  # High school graduate
        'Some college, but degree not received or is in progress': 2,  # Some college
        'Associate\'s degree (for example AA, AS)': 2,  # Some college
        'Bachelor\'s degree (for example BA, BS, AB)': 3,  # Bachelor's degree
        'Graduate degree (for example master\'s, professional, doctorate)': 4  # Graduate degree
    }
    # Apply mapping
    df[var_name] = df[var_name].map(educ_mapping)
    rename_check(df, var_name, new_var_name)

def lucid_recode_education_census(df, var_name='EDUC', new_var_name=None):
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

def lucid_recode_employment_survey(df, var_name='employmentstatus', new_var_name='EMPSTAT'):
    
    accepted_answers = ['Full-time', 'Retired', 'Part-time', 'Unemployed', 'Homemaker', 'Permanently disabled',
                        'Student', 'Other, please specify:', 'Temporarily laid off']
    secondary_accepted_answers = ['self', 'contract'] #some ppl put their answers in "other"
    secondary_variable_name = 'employmentstatus_9_text'
    def encode_employment(value):
        if value == 'Full-time' or value == 'Part-time':
            return 1
        elif value == 'Other, please specify:':
            return 2
        elif value in accepted_answers:
            return 0
        else:
            return None
    
    df[var_name] = df[var_name].apply(encode_employment)

    for i, v in enumerate(df[var_name]):
        if df[var_name][i] != 2:
            continue
        if not isinstance(df[secondary_variable_name][i], str):
            continue
        found_flag = False
        for second_a in secondary_accepted_answers:
            if second_a in df[secondary_variable_name][i].lower():
                df.loc[i,var_name] = 1
                found_flag = True
                break
        if not found_flag:
            df.loc[i,var_name] = None


    rename_check(df, var_name, new_var_name)

def lucid_recode_employment_census(df, var_name='EMPSTAT', new_var_name=None):
    def encode_employment(value):
            if value == 10:
                return 1
            else:
                return 0
            
    df['EMPSTAT'] = df['EMPSTAT'].apply(encode_employment)
    rename_check(df, var_name, new_var_name)
    
def lucid_recode_maritalStatus_survey(df, var_name='maritalstatus', new_var_name='MARST'):
    marital_status_map = {
                    'Now married': 0,  # Married, spouse present -> Now married
                    'Separated': 3,  # Separated -> Separated
                    'Divorced': 2,  # Divorced -> Divorced
                    'Widowed': 1,  # Widowed -> Widowed
                    'Never married': 4   # Never married/single -> Never married
                }
    df[var_name] = df[var_name].map(marital_status_map)
    rename_check(df, var_name, new_var_name)

def lucid_recode_maritalStatus_census(df, var_name='MARST', new_var_name=None):
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

def lucid_recode_race_survey(df, var_name='raceethnicity', new_var_name='RACE'):
    base_race_answers = ['White For example, German, Irish, English, Italian, Polish, French, etc.',
                            'Black or African American For example, African American, Jamaican, Haitian, Nigerian, Ethiopian, Somali, etc.',
                            'Hispanic or Latino For example, Mexican or Mexican American, Puerto Rican, Cuban, Salvadoran, Dominican, Colombian, etc.',
                            'Asian For example, Chinese, Filipino, Asian Indian, Vietnamese, Korean, Japanese, etc.',
                            'American Indian or Alaskan Native For example, Navajo Nation, Blackfeet Tribe, Mayan, Aztec, Native Village of Barrow Inupiat Tribal Government, Tlingit, etc.',
                            'Native Hawaiian or Pacific Islander For example, Native Hawaiian, Samoan, Chamorro, Tongan, Fijian, Marshallese, etc.',
                            'Middle Eastern or North African For example, Lebanese, Iranian, Egyptian, Syrian, Moroccan, Israeli, etc.']

    def encode_race(value):
        num_races = 0
        if not isinstance(value,str):
            return None
        for ba in base_race_answers:
            if ba in value:
                num_races += 1
        if num_races == 0:
            return None
        elif num_races == 1:
            if value == 'White For example, German, Irish, English, Italian, Polish, French, etc.':
                return 0
            elif value == 'Black or African American For example, African American, Jamaican, Haitian, Nigerian, Ethiopian, Somali, etc.':
                return 1
            elif value == 'Asian For example, Chinese, Filipino, Asian Indian, Vietnamese, Korean, Japanese, etc.':
                return 2
            else:
                return 3
        else:
            return 3

    df[var_name] = df[var_name].apply(encode_race)
    rename_check(df, var_name, new_var_name)

def lucid_recode_race_census(df, var_name='RACE', new_var_name=None):
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

def lucid_recode_region_survey(df, var_name='location', new_var_name="REGION"):
    def get_region_id(state_abbr):
        region_mapping = {
            # Region 0: Northeast
            'CT': 0, 'ME': 0, 'MA': 0, 'NH': 0, 'RI': 0, 'VT': 0,
            'NJ': 0, 'NY': 0, 'PA': 0,

            # Region 1: Midwest
            'IL': 1, 'IN': 1, 'MI': 1, 'OH': 1, 'WI': 1,
            'IA': 1, 'KS': 1, 'MN': 1, 'MO': 1, 'NE': 1, 'ND': 1, 'SD': 1,

            # Region 2: South
            'DE': 2, 'DC': 2, 'FL': 2, 'GA': 2, 'MD': 2,
            'NC': 2, 'SC': 2, 'VA': 2, 'WV': 2,
            'AL': 2, 'KY': 2, 'MS': 2, 'TN': 2,
            'AR': 2, 'LA': 2, 'OK': 2, 'TX': 2,

            # Region 3: West
            'AZ': 3, 'CO': 3, 'ID': 3, 'MT': 3, 'NV': 3, 'NM': 3, 'UT': 3, 'WY': 3,
            'AK': 3, 'CA': 3, 'HI': 3, 'OR': 3, 'WA': 3
        }
        return region_mapping.get(state_abbr, None)
    df[var_name] = df[var_name].apply(get_region_id)
    rename_check(df, var_name, new_var_name)

def lucid_recode_region_census(df, var_name='REGION', new_var_name=None):
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

def lucid_recode_sex_survey(df, var_name='sex', new_var_name='SEX'):
    def recode_sex(value):
        if value == 'Female':
            return 1
        elif value == 'Male':
            return 0
        else:
            return None
    df[var_name] = df[var_name].apply(recode_sex)
    rename_check(df, var_name, new_var_name)
            
def lucid_recode_sex_census(df, var_name='SEX', new_var_name=None):
    sex_mapping = {
    1: 0, #male
    2: 1, #female
    }
    df[var_name] = df[var_name].map(sex_mapping)
    rename_check(df, var_name, new_var_name)

def lucid_recode_survey(survey_df, year):
    lucid_recode_age_general(survey_df)
    lucid_recode_education_survey(survey_df)
    lucid_recode_maritalStatus_survey(survey_df)
    lucid_recode_race_survey(survey_df)
    lucid_recode_region_survey(survey_df)
    lucid_recode_sex_survey(survey_df)
    lucid_recode_employment_survey(survey_df)

def lucid_recode_census(census_df, year):
    lucid_recode_age_general(census_df,var_name='AGE')
    lucid_recode_education_census(census_df)
    lucid_recode_maritalStatus_census(census_df)
    lucid_recode_race_census(census_df)
    lucid_recode_region_census(census_df)
    lucid_recode_sex_census(census_df)
    lucid_recode_employment_census(census_df)

def recoding_survey_and_census_data(survey_df, census_df, target_var):

    '''
    survey_df - pandas df - household census survey data
    census_df - pandas df - IPUMS survey 
    target_var - list[str] - list of target variables
    iterates over a survey df and census df and re-encodes all variables manually

    reincoding code was generated by CHATGPT
    '''
    #drop all columns not prevelant to each survey
    relevant_columns = ['AGE', 'REGION', 'EDUC', 'SEX', 'MARST', 'RACE']
    year = 2023
    if survey_df is not None:
        lucid_recode_survey(survey_df, year)
        survey_df = survey_df.filter(items=target_var + relevant_columns)
        survey_df = survey_df.dropna()

    if census_df is not None:
        lucid_recode_census(census_df, year)
        census_df.rename(columns={'WTFINL':'PERWT'},inplace=True)
        census_df['PERWT'] = census_df['PERWT']
        census_df = census_df.filter(items=['PERWT'] + relevant_columns)
        census_df = census_df.dropna()

    return survey_df, census_df
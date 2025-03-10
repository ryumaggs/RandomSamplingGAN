import pandas as pd 

read_file = "ipums_dataset/raw/ipums.csv"
save_file = "ipums_dataset/cleaned/ipums.csv"

variable = ['AGE', 
            'SEX', 
            'HISPAN', 
            'RACE', 
            'EDUC', 
            'MARST', 
            'WRKLSTWK', 
            'FTOTINC', 
#            'MIGPLAC1', 
            'HCOVANY']
new_variable = ['Age',
            'Gender', 
            'Hispan', 
            'Race', 
            'Education', 
            'Marital', 
            'Work', 
            'Income', 
#            'Region', 
            'Insurance']

df = pd.read_csv(read_file)[variable]

# Drop missing value
df = df.dropna()
df = df[~df['SEX'].isin([9])]
df['EDUC'] = df['EDUC'].astype(int)
df = df[(df['EDUC'] >= 1) & (df['EDUC'] <= 11)]
df = df[~df['WRKLSTWK'].isin([0,3])]
df['FTOTINC'] = df['FTOTINC'].astype(int)
df = df[(df['FTOTINC'] >= 2) & (df['FTOTINC'] <= 9999997)]
#df['MIGPLAC1'] = df['MIGPLAC1'].astype(int)
#df = df[(df['MIGPLAC1'] >= 1) & (df['MIGPLAC1'] <= 56)]


education_mapping = {
    1: 1,  # Nursery school to grade 4 -> Less than high school
    2: 1,  # Grade 5, 6, 7, or 8 -> Less than high school
    3: 2,  # Grade 9 -> Some high school
    4: 2,  # Grade 10 -> Some high school
    5: 2,  # Grade 11 -> Some high school
    6: 3,  # Grade 12 -> High school graduate
    7: 4,  # 1 year of college -> Some college
    8: 4,  # 2 years of college -> Some college
    9: 4,  # 3 years of college -> Some college
    10: 6,  # 4 years of college -> Bachelor's degree
    11: 7,  # 5+ years of college -> Graduate degree
}


df['AGE'] = df['AGE']                                                                   # Age
df['SEX'] = df['SEX'] - 1                                                               # Sex
df['HISPAN'] = df['HISPAN'].apply(lambda x: 1 if x in [1, 2, 3, 4] else x)              # Hispanic
df['RACE'] = df['RACE'].replace({2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4})       # Race             
df['EDUC'] = df['EDUC'].map(education_mapping)                                          # Education                          
df['MARST'] = df['MARST'].replace({1: 1, 2: 1, 3: 4, 4: 3, 5: 2, 6: 5})                 # Marital status                         
df['WRKLSTWK'] = df['WRKLSTWK'] - 1                                                     # Work
df['FTOTINC'] = pd.cut(
    df['FTOTINC'],
    bins=[0, 24999, 34999, 49999, 74999, 99999, 149999, 199999, float('inf')],
    labels=[1, 2, 3, 4, 5, 6, 7, 8],
    right=True
)                                                                                   # Income
df['HCOVANY'] = df['HCOVANY'] - 1                                                   # Insurance

# Rename
df.rename(columns=dict(zip(variable, new_variable)), inplace=True)

# One-hot encoding
df = df[df['Age'] >= 18]
bins = [18, 34, 49, 64, 84, float('inf')]
labels = ['18-34', '35-49', '50-64', '65-84', '85+']
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
df = pd.get_dummies(df, columns=['Age'], prefix='Age', drop_first=False)
df = pd.get_dummies(df, columns=['Race'], prefix='Race', drop_first=False)
fake_row = pd.DataFrame({'Education': [5]})                         # Because "Education" = 5 is rare,
df = pd.concat([df, fake_row], ignore_index=True)                   # we need to add a fake datapoint 
df = pd.get_dummies(df, columns=['Education'], prefix='Education')  # to ensure there is no missing column
df = df.iloc[:-1]
df = pd.get_dummies(df, columns=['Marital'], prefix='Marital', drop_first=False)
df = pd.get_dummies(df, columns=['Income'], prefix='Income', drop_first=False)
df = df.astype('float64')

#df = pd.get_dummies(df, columns=['Region'], prefix='Region')                    # State

print(df.shape[0])
print(df.head(2))
df = df.sample(n=10000, random_state=42)
df.to_csv(save_file, index=False)

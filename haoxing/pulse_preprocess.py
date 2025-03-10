import pandas as pd

# Read the CSV file
read_files = ["pulse_dataset/raw/week22/pulse2021_puf_22.csv", 
        "pulse_dataset/raw/week23/pulse2021_puf_23.csv", 
        "pulse_dataset/raw/week24/pulse2021_puf_24.csv", 
        "pulse_dataset/raw/week25/pulse2021_puf_25.csv", 
        "pulse_dataset/raw/week26/pulse2021_puf_26.csv", 
        "pulse_dataset/raw/week27/pulse2021_puf_27.csv", 
        "pulse_dataset/raw/week28/pulse2021_puf_28.csv", 
        "pulse_dataset/raw/week29/pulse2021_puf_29.csv"]
save_files = ["pulse_dataset/cleaned/week22/week22.csv", 
        "pulse_dataset/cleaned/week23/week23.csv", 
        "pulse_dataset/cleaned/week24/week24.csv", 
        "pulse_dataset/cleaned/week25/week25.csv", 
        "pulse_dataset/cleaned/week26/week26.csv", 
        "pulse_dataset/cleaned/week27/week27.csv", 
        "pulse_dataset/cleaned/week28/week28.csv", 
        "pulse_dataset/cleaned/week29/week29.csv"]

variable = ['TBIRTH_YEAR', 
            'EGENDER', 
            'RHISPANIC', 
            'RRACE', 
            'EEDUC', 
            'MS', 
            'ANYWORK', 
            'INCOME', 
#            'EST_ST', 
            'PUBHLTH', 
            'RECVDVACC']

new_variable = ['Age',
            'Gender', 
            'Hispan', 
            'Race', 
            'Education', 
            'Marital', 
            'Work', 
            'Income', 
#            'Region', 
            'Insurance', 
            'Vaccine']

for read_file, save_file in zip(read_files, save_files):

    df = pd.read_csv(read_file)[variable]
    

    # Drop missing value
    df = df.dropna()
    df = df[~df['MS'].isin([-99, -88])]
    df = df[~df['INCOME'].isin([-99, -88])]
    df = df[~df['PUBHLTH'].isin([3])]
    df = df[~df['RECVDVACC'].isin([-99, -88])]
    
    df['TBIRTH_YEAR'] = 2021 - df['TBIRTH_YEAR']                 # Year
    df = df[df['TBIRTH_YEAR'] >= 18]
    df['EGENDER'] = df['EGENDER'] - 1                            # Sex: from 1/2 to 0/1
    df['RHISPANIC'] = df['RHISPANIC'] - 1                        # Hispanic: from 1/2 to 0/1
    df['RRACE'] = df['RRACE']                                    # Race:
    df['EEDUC'] = df['EEDUC']                                    # Education
    df['MS'] = df['MS']                                          # Marital status
    df['ANYWORK'] = 2 - df['ANYWORK']                            # Work: from 1/2 to 1/0
    df['INCOME'] = df['INCOME']                                  # Income
    df['PUBHLTH'] = 2 - df['PUBHLTH']                            # Insurance: from 1/2 to 1/0
    df['RECVDVACC'] = 2 - df['RECVDVACC']                        # Vaccine: from 1/2 to 1/0

    # Rename
    df.rename(columns=dict(zip(variable, new_variable)), inplace=True)

    # One-hot encoding
    df = df[df['Age'] >= 18]
    bins = [18, 34, 49, 64, 84, float('inf')]
    labels = ['18-34', '35-49', '50-64', '65-84', '85+']
    df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    df = pd.get_dummies(df, columns=['Age'], prefix='Age')
    df = pd.get_dummies(df, columns=['Race'], prefix='Race')
    df = pd.get_dummies(df, columns=['Education'], prefix='Education')
    df = pd.get_dummies(df, columns=['Marital'], prefix='Marital')
    df = pd.get_dummies(df, columns=['Income'], prefix='Income')
    df = df.astype('float64')


# Convert Age to categorical variable


    #df = pd.get_dummies(df, columns=['Region'], prefix='Region') # Region

    df = df[[col for col in df.columns if col != 'Vaccine'] + ['Vaccine']]
    print(df.shape[0])
    print(df.head(2))
    df = df.sample(n=2500, random_state=42)
    print(df["Vaccine"].mean())
    df.to_csv(save_file, index=False)

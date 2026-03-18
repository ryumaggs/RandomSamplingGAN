import weightipy as wp
import pandas as pd
import numpy as np
if __name__ == "__main__":
    #load survey data
    for i in range(20,26):
        week = str(i)
        survey_path = './data/progress_data/cleaned/d4p_week' + week + '_cleaned.csv'
        ground_truth_path = './data/progress_data/cleaned/ipums_cleaned_combined.csv'
        
        surv = pd.read_csv(survey_path)
        gtd = pd.read_csv(ground_truth_path)
        #load census data

        # build weighted targets from gtd (weighted by PERWT, excluding PERWT itself)
        total_weight = gtd['PERWT'].sum()
        targets = {}
        for col in gtd.columns:
            if col == 'PERWT':
                continue
            targets[col] = (
                gtd.groupby(col)['PERWT'].sum() / total_weight * 100
            ).to_dict()

        #print(targets)
        
        '''
        #obtain census targets in the form of:
        targets = {
        "age_group": {"18-24": 10.0, "25+": 90.0},
        "gender": {"Male": 49.0, "Female": 51.0}
        }
        '''

        scheme = wp.scheme_from_dict(targets)
        df_weighted = wp.weight_dataframe(surv, scheme, weight_column="raking_weights")

        vacc = np.expand_dims(df_weighted['RECVDVACC'].to_numpy(),0)
        raking_weights = np.expand_dims(df_weighted['raking_weights'].to_numpy(),1) / np.sum(df_weighted['raking_weights'].to_numpy())

        print(week + ", prediction: ", (vacc @ raking_weights).item())
import weightipy as wp
import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np

def compare_dataframes(dfs, names=None):
    """
    Compare and contrast a list of pandas DataFrames with categorical (int-encoded) variables.
    
    Args:
        dfs: list of pandas DataFrames
        names: optional list of names for each df (e.g. ['survey', 'gtd', 'synthetic'])
    """
    if names is None:
        names = [f"df_{i}" for i in range(len(dfs))]

    columns = dfs[0].columns.tolist()

    # --- 1. Shape comparison ---
    print("=== Shape ===")
    for name, df in zip(names, dfs):
        print(f"  {name}: {df.shape}")

    # --- 2. Per-variable value distribution comparison ---
    print("\n=== Value Distribution (proportions) per variable ===")
    for col in columns:
        print(f"\n  [{col}]")
        all_vals = sorted(set(v for df in dfs for v in df[col].unique()))
        rows = []
        for name, df in zip(names, dfs):
            counts = df[col].value_counts(normalize=True).reindex(all_vals, fill_value=0)
            rows.append(counts.rename(name))
        dist_df = pd.DataFrame(rows).T
        dist_df.index.name = "value"
        print(dist_df.round(4).to_string())

    # --- 3. Per-variable mean and std ---
    print("\n=== Mean and Std per variable ===")
    stats = {}
    for name, df in zip(names, dfs):
        stats[name] = df.agg(["mean", "std"]).T
    summary = pd.concat(stats, axis=1)
    print(summary.round(4).to_string())

    if False:
        # --- 4. Pairwise TVD (Total Variation Distance) per variable ---
        print("\n=== Pairwise Total Variation Distance (TVD) per variable ===")
        tvd_results = {}
        for i in range(len(dfs)):
            for j in range(i + 1, len(dfs)):
                pair = f"{names[i]} vs {names[j]}"
                tvds = {}
                for col in columns:
                    all_vals = sorted(set(dfs[i][col].unique()) | set(dfs[j][col].unique()))
                    p = dfs[i][col].value_counts(normalize=True).reindex(all_vals, fill_value=0)
                    q = dfs[j][col].value_counts(normalize=True).reindex(all_vals, fill_value=0)
                    tvds[col] = 0.5 * np.abs(p - q).sum()
                tvd_results[pair] = tvds

        tvd_df = pd.DataFrame(tvd_results).round(4)
        print(tvd_df.to_string())

        return tvd_df


if __name__ == "__main__":
    #load survey data
    all_surveys = []
    all_names = []
    for i in range(35,46):
        week = str(i)
        #survey_path = './data/progress_data/cleaned/d4p_week' + week + '_cleaned.csv'
        #survey_path = './data/censusHouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv'
        survey_path = f"./data/axios_ipsos_data/cleaned/week{week}_cleaned.csv"
        ground_truth_path = './data/progress_data/cleaned/ipums_cleaned_combined.csv'
         

        if not os.path.exists(survey_path):
            continue


        surv = pd.read_csv(survey_path)
        gtd = pd.read_csv(ground_truth_path)

        all_surveys.append(surv)
        all_names.append(week)

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

    if False:
        ground_truth_path = './data/progress_data/cleaned/ipums_cleaned_combined.csv'
        gtd = pd.read_csv(ground_truth_path)
        gtd.rename(columns={"PERWT": "RECVDVACC"}, inplace=True)
        all_surveys.append(gtd)
        all_names.append('census')
        compare_dataframes(all_surveys, names=all_names)

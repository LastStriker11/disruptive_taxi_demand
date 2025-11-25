import pandas as pd
import numpy as np
import os
import random

def def_od_matrix():
    od_list = []
    for i in range(1,78):
        for j in range(1,78):
            od_list.append((i,j))
    return od_list

def stable_period_normalize(df, df_filtered, stable_start, stable_end):
    stable_mask = (df['Trip Start Timestamp'] >= stable_start) & (df_filtered['Trip Start Timestamp'] < stable_end)
    df_stable = df_filtered.loc[stable_mask].copy()
    # Group by weekday and get average daily demand
    stable_avg = (
        df_stable.groupby(['Pickup Community Area', 'Dropoff Community Area', 'weekday'])
                .size()
                .reset_index(name='avg_demand')
    )
    # Demand normalization
    df_daily = (
        df_filtered.groupby(['Pickup Community Area', 'Dropoff Community Area', 'date', 'weekday'])
                .size()
                .reset_index(name='num_trips')
    )
    df_normalized = pd.merge(
        df_daily,
        stable_avg,
        on=['Pickup Community Area', 'Dropoff Community Area', 'weekday'],
        how='left'
    )
    # Normalize
    df_normalized['normalized_demand'] = 2 * df_normalized['num_trips'] / df_normalized['avg_demand']
    return df_normalized

def od_normalized_demand(df_normalized, start_day, end_day):
    od_list = def_od_matrix()
    od_normalized = []
    date_list = sorted(df_normalized['date'].unique())
    for i in range(0,len(od_list)):
        pair_df = df_normalized[
            (df_normalized['Pickup Community Area'] == od_list[i][0]) &
            (df_normalized['Dropoff Community Area'] == od_list[i][1])
        ]
        pair_df = pair_df.set_index('date').reindex(date_list, fill_value=0).reset_index()
        od_normalized.extend(pair_df['normalized_demand'].tolist())
    s = int(len(od_normalized)/len(date_list))
    od_normalized = np.asmatrix(od_normalized).reshape(s,len(date_list))
    od_normalized = np.array(od_normalized)
    # select specific days
    od_normalized = od_normalized[:,start_day:end_day]
    df_od = pd.DataFrame(od_normalized, columns = [i for i in range(start_day,end_day)])
    df_od.index = od_list
    return df_od

def seed_everything(seed: int = 42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"Randomness fixed with seed={seed}")

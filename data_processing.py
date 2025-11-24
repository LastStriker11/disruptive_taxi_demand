#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from src.miscs import def_od_matrix, stable_period_normalize, od_normalized_demand
#%%
# COVID period data: weekly summarized taxi trips between community areas
year = 2022
df = pd.read_csv(f'Taxi_Trips_{year}.csv',usecols=['Trip Start Timestamp','Trip Seconds','Trip Miles','Pickup Community Area' ,'Dropoff Community Area'])

df.dropna(subset = ['Trip Start Timestamp','Pickup Community Area', 'Dropoff Community Area'],
          axis = 0,
          how = 'any',
          inplace = True
          )

df["start"] = df["Trip Start Timestamp"].map(lambda x:x.split(" ")[0])
df["start"] = pd.to_datetime(df["start"], format='%m/%d/%Y')
df["number of weeks"] = df["start"].dt.isocalendar().week
df = df[['Trip Seconds',"Trip Miles","Pickup Community Area","Dropoff Community Area","start","number of weeks"]]

df["year"] = df["start"].dt.isocalendar().year
df["num_weeks"]=df["number of weeks"]+(df["year"]-2019)*52

od_list = def_od_matrix()

odtrips = []
w_min = df['num_weeks'].min()
w_max = df['num_weeks'].max()
for i in range(0,len(od_list)):   
    df_area = df[(df["Pickup Community Area"] == od_list[i][0])&(df["Dropoff Community Area"]==od_list[i][1])]
    count_area = df_area['num_weeks'].value_counts()
    for j in range(w_min, w_max+1):
        if j in count_area.index:
            odtrips.append(count_area[j])
        else:
            odtrips.append(0)

s = int(len(odtrips)/(w_max - w_min + 1))
odtrips = np.asmatrix(odtrips).reshape(s,w_max - w_min + 1)
odtrips = np.array(odtrips)

df_od = pd.DataFrame(odtrips, columns = [i for i in range(w_min,w_max+1)])
df_od.index = od_list
df_od.to_csv(f'results/taxi_od_weekly_counts_{year}.csv')
#%%
# filter od pairs
df = pd.DataFrame()
for y in [2019,2020,2021,2022]:
    df_year = pd.read_csv(f'results/taxi_od_weekly_counts_{y}.csv', index_col=0)
    df = pd.concat([df,df_year], axis=1)
df.columns = [f"week_{i}" for i in range(1,df.shape[1]+1)]
df.fillna(0, inplace=True)

# filtering OD pairs with very few trips across all weeks can be done here
df = df.iloc[:, 55:95] # focusing on weeks 56 to 95 (from March 2020 to October 2020)
num_filters = 500
df = df[(df.sum(axis=1) > num_filters)]
# save od pairs after filtering
df.to_csv('results/covid_od_filtered.csv')

# normalize data using the average of the first 5 weeks (pre-covid)
w_max = 5
odtrips = df.to_numpy()
row_avg = odtrips[:, :w_max].mean(axis=1, keepdims=True)
odtrips = odtrips / row_avg
np.save('results/covid_demand_normalized.npy', odtrips)
#%%
pickup_area = 32
dropoff_area = 8

pair_df = odtrips[0,:]

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(range(len(pair_df)), pair_df, marker='o', linestyle='-')
ax.set_xlabel('Time (weeks)')
ax.set_ylabel('Normalized Demand')
ax.grid(True)
#%%
# Chicago snowstorm period data: daily summarized taxi trips between community areas
df = pd.read_csv('data/Taxi_Trips_2015.csv',usecols=['Trip Start Timestamp','Trip Seconds','Trip Miles','Pickup Community Area' ,'Dropoff Community Area'])
df.dropna(subset=['Trip Start Timestamp','Pickup Community Area', 'Dropoff Community Area'], axis=0, how='any', inplace=True)

# Ensure timestamp is datetime
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')

# Filter the date range
start_date = '2015-11-01'
end_date = '2015-12-31'

mask = (df['Trip Start Timestamp'] >= start_date) & (df['Trip Start Timestamp'] < end_date)
df_filtered = df.loc[mask].copy()

# Create hourly timestamp for grouping
df_filtered['day_id'] = df_filtered['Trip Start Timestamp'].dt.floor('D').dt.dayofyear

df_filtered['date'] = df_filtered['Trip Start Timestamp'].dt.date
df_filtered['weekday'] = df_filtered['Trip Start Timestamp'].dt.day_name()
#%%
# Define stable period for average demand calculation
stable_start = '2015-11-01'
stable_end = '2015-11-15'
# df_normalized = stable_period_normalize(df, df_filtered, stable_start, stable_end)
stable_mask = (df['Trip Start Timestamp'] >= stable_start) & (df_filtered['Trip Start Timestamp'] < stable_end)
df_stable = df_filtered.loc[stable_mask].copy()
#%%
# Group by weekday and get average daily demand
stable_avg = (
    df_stable.groupby(['Pickup Community Area', 'Dropoff Community Area', 'weekday'])
            .size()
            .reset_index(name='avg_demand')
)
# stable_avg = stable_avg[stable_avg['avg_demand']>500] # filter ODs
# Demand normalization
df_daily = (
    df_filtered.groupby(['Pickup Community Area', 'Dropoff Community Area', 'date', 'weekday'])
            .size()
            .reset_index(name='num_trips')
)
df_daily = df_daily[df_daily['num_trips']>10]
df_normalized = pd.merge(
    df_daily,
    stable_avg,
    on=['Pickup Community Area', 'Dropoff Community Area', 'weekday'],
    how='left'
)
df_normalized.dropna(inplace=True)
# Normalize
df_normalized['normalized_demand'] = 2 * df_normalized['num_trips'] / df_normalized['avg_demand']
#%%
# visualization for a specific OD pair
pickup_area = 32
dropoff_area = 8

pair_df = df_normalized[
    (df_normalized['Pickup Community Area'] == pickup_area) &
    (df_normalized['Dropoff Community Area'] == dropoff_area)
].sort_values('date')

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(pair_df['date'], pair_df['normalized_demand'], marker='o', linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Demand')
ax.grid(True)
# %%
# save OD daily normalized demand to csv
df_od = od_normalized_demand(df_normalized, start_day=15, end_day=29)  # from 2015-11-16 to 2015-11-29, total 14 days
od_matrix = df_od.values
# Threshold
threshold = 0.3
# Compute proportion of zeros per row
zero_fraction = np.mean(od_matrix == 0, axis=1)
mask = zero_fraction <= threshold
od_matrix = od_matrix[mask]
kept_indices = np.where(mask)[0]
np.save('results/snowstorm_demand_normalized_v2.npy', od_matrix)
np.save('results/snowstorm_od_index_v2.npy', kept_indices)
# # load filtered OD pairs
# od_filtered = pd.read_csv('results/covid_od_filtered.csv', index_col=0).index.map(ast.literal_eval)
# # filter normalized OD matrix
# od_normalized_filtered = df_od.loc[od_filtered]
# np.save('results/snowstorm_demand_normalized.npy', od_normalized_filtered.values)
# %%
# Define stable period for average demand calculation before Christmas
stable_start = '2015-12-01'
stable_end = '2015-12-15'
# df_normalized = stable_period_normalize(df, df_filtered, stable_start, stable_end)
stable_mask = (df['Trip Start Timestamp'] >= stable_start) & (df_filtered['Trip Start Timestamp'] < stable_end)
df_stable = df_filtered.loc[stable_mask].copy()
#%%
# Group by weekday and get average daily demand
stable_avg = (
    df_stable.groupby(['Pickup Community Area', 'Dropoff Community Area', 'weekday'])
            .size()
            .reset_index(name='avg_demand')
)
# stable_avg = stable_avg[stable_avg['avg_demand']>500] # filter ODs
# Demand normalization
df_daily = (
    df_filtered.groupby(['Pickup Community Area', 'Dropoff Community Area', 'date', 'weekday'])
            .size()
            .reset_index(name='num_trips')
)
df_daily = df_daily[df_daily['num_trips']>10]
df_normalized = pd.merge(
    df_daily,
    stable_avg,
    on=['Pickup Community Area', 'Dropoff Community Area', 'weekday'],
    how='left'
)
df_normalized.dropna(inplace=True)
# Normalize
df_normalized['normalized_demand'] = 2 * df_normalized['num_trips'] / df_normalized['avg_demand']
# %%
# save OD daily normalized demand to csv
df_od = od_normalized_demand(df_normalized, start_day=43, end_day=57)  # from 2015-12-14 to 2015-12-27, total 14 days
od_matrix = df_od.values
kept_indices = np.load('results/snowstorm_od_index_v2.npy')
od_matrix = od_matrix[kept_indices]
np.save('results/christmas_demand_normalized_v2.npy', od_matrix)
# filter normalized OD matrix
# od_normalized_filtered = df_od.loc[od_filtered]
# np.save('results/christmas_demand_normalized.npy', od_normalized_filtered.values)
# %%
# visualization for a specific OD pair
pickup_area = 32
dropoff_area = 8

pair_df = df_od[df_od.index == (pickup_area,dropoff_area)].values[0]

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(range(len(pair_df)), pair_df, marker='o', linestyle='-')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Normalized Demand')
ax.grid(True)
# %%
# Chicago Christmas period data (2018)
df = pd.read_csv('data/Taxi_Trips_2018.csv',usecols=['Trip Start Timestamp','Trip Seconds','Trip Miles','Pickup Community Area' ,'Dropoff Community Area'])
df.dropna(subset=['Trip Start Timestamp','Pickup Community Area', 'Dropoff Community Area'], axis=0, how='any', inplace=True)

# Ensure timestamp is datetime
df['Trip Start Timestamp'] = pd.to_datetime(df['Trip Start Timestamp'], errors='coerce')

# Filter the date range
start_date = '2018-11-01'
end_date = '2018-12-31'

mask = (df['Trip Start Timestamp'] >= start_date) & (df['Trip Start Timestamp'] < end_date)
df_filtered = df.loc[mask].copy()

# Create hourly timestamp for grouping
df_filtered['day_id'] = df_filtered['Trip Start Timestamp'].dt.floor('D').dt.dayofyear

df_filtered['date'] = df_filtered['Trip Start Timestamp'].dt.date
df_filtered['weekday'] = df_filtered['Trip Start Timestamp'].dt.day_name()
#%%
stable_start = '2018-12-01'
stable_end = '2018-12-15'
# df_normalized = stable_period_normalize(df, df_filtered, stable_start, stable_end)
stable_mask = (df['Trip Start Timestamp'] >= stable_start) & (df_filtered['Trip Start Timestamp'] < stable_end)
df_stable = df_filtered.loc[stable_mask].copy()

# Group by weekday and get average daily demand
stable_avg = (
    df_stable.groupby(['Pickup Community Area', 'Dropoff Community Area', 'weekday'])
            .size()
            .reset_index(name='avg_demand')
)
# stable_avg = stable_avg[stable_avg['avg_demand']>500] # filter ODs
# Demand normalization
df_daily = (
    df_filtered.groupby(['Pickup Community Area', 'Dropoff Community Area', 'date', 'weekday'])
            .size()
            .reset_index(name='num_trips')
)
df_daily = df_daily[df_daily['num_trips']>10]
df_normalized = pd.merge(
    df_daily,
    stable_avg,
    on=['Pickup Community Area', 'Dropoff Community Area', 'weekday'],
    how='left'
)
df_normalized.dropna(inplace=True)
# Normalize
df_normalized['normalized_demand'] = 2 * df_normalized['num_trips'] / df_normalized['avg_demand']
#%%
# visualization for a specific OD pair
pickup_area = 32
dropoff_area = 8

pair_df = df_normalized[
    (df_normalized['Pickup Community Area'] == pickup_area) &
    (df_normalized['Dropoff Community Area'] == dropoff_area)
].sort_values('date')

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(pair_df['date'], pair_df['normalized_demand'], marker='o', linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Demand')
ax.grid(True)
# %%
# save OD daily normalized demand to csv
df_od = od_normalized_demand(df_normalized, start_day=43, end_day=57)  # from 12-14 to 12-26, total 14 days
od_matrix = df_od.values
kept_indices = np.load('results/snowstorm_od_index_v2.npy')
od_matrix = od_matrix[kept_indices]
np.save('results/christmas_demand_normalized_2018.npy', od_matrix)
# %%

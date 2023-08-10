import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# load data from csv file
DRL_Data = pd.DataFrame(pd.read_csv('wandb_export_2023-08-08T12_36_11.018-04_00.csv'))
# only keep first two columns
DRL_Data = DRL_Data.iloc[:, :2]

# load data from csv file
BP_Data = pd.DataFrame(pd.read_csv("wandb_export_2023-08-08T12_36_47.772-04_00.csv"))
# only keep first two columns
BP_Data = BP_Data.iloc[:, :2]

# get series of the union of the first column in each dataframe
x = pd.Series(np.arange(min(BP_Data.iloc[:, 0]), max(DRL_Data.iloc[:, 0]) + 1))
# create a new data frame with the first column as x and the other columns being equivalent to df[x] for each df
DRL_Data = pd.DataFrame(x).merge(DRL_Data, how='left', left_on=0, right_on='Step')
# remove 'Step' column
DRL_Data = DRL_Data.drop(columns=['Step'])
# interpolate missing values
DRL_Data = DRL_Data.interpolate()

# do the same for other dataframe
BP_Data = pd.DataFrame(x).merge(BP_Data, how='left', left_on=0, right_on='Step')
BP_Data = BP_Data.drop(columns=['Step'])
BP_Data = BP_Data.interpolate()

# Merge DRL Data and BP Data
Data = pd.concat([DRL_Data, BP_Data], axis=1)
Data.columns = ['StepDRL', 'DRL', 'StepBP', 'BP']
# drop StepBP column
Data = Data.drop(columns=['StepBP'])

# plot data
# fig, ax = plt.subplots()
# ax.plot(Data['StepDRL'], Data['DRL'], label='IAPPO')
# ax.plot(Data['StepDRL'], Data['BP'], label='BP')
# ax.set_xlabel('Time Step')
# ax.set_ylabel('LTA Backlog')
# ax.set_title('LTA Backlog vs Time Step')
# ax.legend()
# plt.show()

# Smoothed average reward PPO Data
IAPPO_SA_Data = pd.DataFrame(pd.read_csv('wandb_export_2023-08-08T13_14_15.848-04_00.csv'))
IAPPO_SA_Data = IAPPO_SA_Data.iloc[:, 3:5]
IAPPO_SA_Data.columns = ['Step', 'IAPPO']
IAPPO_SA_Data= pd.DataFrame(x).merge(IAPPO_SA_Data, how='left', left_on=0, right_on='Step')
# remove 'Step' column
IAPPO_SA_Data = IAPPO_SA_Data.drop(columns=['Step'])
# interpolate missing values
IAPPO_SA_Data = IAPPO_SA_Data.interpolate()
# rename columns
Data2 = pd.concat([Data, IAPPO_SA_Data], axis=1)
#drop StepIAPPO column

# plot data
fig, ax = plt.subplots()
ax.plot(Data2['StepDRL'], Data2['DRL'], label='IAPPO (LTA)')
ax.plot(Data2['StepDRL'], Data2['BP'], label='BP (LTA)')
ax.plot(Data2['StepDRL'], Data2['IAPPO'], label='IAPPO (SA[1000])')
ax.set_xlabel('Time Step')
ax.set_ylabel('Backlog')
ax.set_title('Backlog vs Time Step')
ax.legend()
#plt.show()

# import intervention rate data
Intervention_Data = pd.DataFrame(pd.read_csv('IAPPO_InterventionRate.csv'))
Intervention_Data.drop(Intervention_Data.columns[0], axis=1, inplace=True)
fig, ax = plt.subplots()
ax.plot(Intervention_Data.index, Intervention_Data['Intervention Rate'], label = 'IAPPO Intervention Rate')
ax.set_xlabel("Episode")
ax.set_ylabel("Intervention Rate")
ax.set_title("Intervention Rate vs Episode")
ax.legend()
plt.show()
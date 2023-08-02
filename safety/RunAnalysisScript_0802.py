import pickle
import numpy as np
import pandas
from safety.loggers import *
import matplotlib.pyplot as plt
import os

th = pickle.load(open("JRQ_Test_History_0802.pkl", "rb"))
th = pickle.load(open("JSQ_Test_History_0802.pkl", "rb"))

JSQ_Mean = 4.8
JRQ_Mean = 14.98
save_path = "Analysis2"
if not os.path.exists(save_path):
    os.mkdir(save_path)

SAVE = True
SHOW = True
# Plot Backlogs vs Time
fig, ax = plt.subplots()
ax.plot(th["backlogs"])
plt.title("Backlogs vs Time")
plt.xlabel("Time")
plt.ylabel("Backlogs")
plt.ylim([0, 50])
if SAVE: fig.savefig(f"{save_path}/Backlogs_vs_Time.png")
if SHOW: fig.show()


# find the first time the LTA Backlog is close to the JRQ Mean


# Plot LTA Rewards vs Time
fig, ax = plt.subplots()
ax.plot(th["LTA_Backlogs"])
plt.title("Long-Term Average (LTA) Backlog vs Time")
plt.xlabel("Time")
plt.ylabel("LTA Backlog")
plt.ylim([0, 20])
if SAVE: fig.savefig(f"{save_path}/LTA_Backlogs_vs_Time_zoomed.png")
if SHOW: fig.show()

fig, ax = plt.subplots()
ax.plot(th["LTA_Backlogs"])
plt.title("Long-Term Average (LTA) Backlog vs Time")
plt.xlabel("Time")
plt.ylabel("LTA Backlog")
plt.ylim([0, 50])
if SAVE: fig.savefig(f"{save_path}/LTA_Backlogs_vs_Time.png")
if SHOW:   fig.show()

# Get Rolling Average Backlogs
for w in [500, 1000, 2000, 5000, 10000, 20000]:
    roll_avg_backlogs = pd.Series(th["backlogs"][:,0]).rolling(w, min_periods=w).mean().values

    fig, ax = plt.subplots()
    ax.plot(roll_avg_backlogs, label="Rolling Average Backlogs")
    ax.plot([JRQ_Mean] * len(roll_avg_backlogs), linestyle = "--", color = "blue", label = "JRQ Mean")
    ax.plot([JSQ_Mean] * len(roll_avg_backlogs), linestyle = "--", color = "red", label = "JSQ Mean")
    plt.title(f"Rolling Average Backlogs (w = {w}) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Rolling Average Backlogs")
    plt.ylim([0, 50])
    plt.legend()
    if SAVE: fig.savefig(f"{save_path}/Rolling_Average_Backlogs_vs_Time_w_{w}.png")
    if SHOW: fig.show()

# Get Rolling Standard Deviation Backlogs
for w in [500, 1000, 2000, 5000, 10000, 20000]:
    roll_std_backlogs = pd.Series(th["backlogs"][:,0]).rolling(w, min_periods=w).std().values

    fig, ax = plt.subplots()
    ax.plot(roll_std_backlogs, label="Rolling Std Backlogs")
    # ax.plot([JRQ_Mean] * len(roll_avg_backlogs), linestyle = "--", color = "blue", label = "JRQ Mean")
    # ax.plot([JSQ_Mean] * len(roll_avg_backlogs), linestyle = "--", color = "red", label = "JSQ Mean")
    plt.title(f"Rolling Std Backlogs (w = {w}) vs Time")
    plt.xlabel("Time")
    plt.ylabel("Rolling Std Backlogs")
    plt.ylim([0, 20])
    plt.legend()
    if SAVE: fig.savefig(f"{save_path}/Rolling_Std_Backlogs_vs_Time_w_{w}.png")
    if SHOW: fig.show()

# Save all open plots

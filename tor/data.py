# %%

import numpy as np 
import pandas as pd 
import xgboost
from sklearn import model_selection, metrics

# %%
def load_and_prep_data():
    df = pd.read_csv("input/A-5s.csv")
    df.drop(df.index[-39:], inplace=True)
    df.replace('Infinity', -1, inplace=True)
    df["label"] = df["label"].map({"nonTOR": 0, "TOR": 1})
    df["Source IP"] = df["Source IP"].apply(lambda x: float(x.replace(".", "")))
    df[" Destination IP"] = df[" Destination IP"].apply(lambda x: float(x.replace(".", "")))
    return df

# %%

df = load_and_prep_data()
features = [
       ' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
       'Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
       'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
       ' Idle Std', ' Idle Max', ' Idle Min',]

X = np.array(df[features])
y = np.array(df["label"])
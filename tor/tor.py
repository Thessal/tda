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
print(X.shape, np.mean(y))

#%%
splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in splitter.split(X, y):
    X_train, X_holdout = X[train_idx], X[test_idx]
    y_train, y_holdout = y[train_idx], y[test_idx]
    
#%%
model = xgboost.XGBClassifier(seed=42)
skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_test_fold = X_train[test_idx]
    y_test_fold = y_train[test_idx]
    model.fit(X_train_fold, y_train_fold)
    probas = model.predict_proba(X_test_fold)[:, 1]
    preds = (probas > 0.5).astype(int)
    #y_pred = model.predict(X_val_folds)
    #print(f"Fold {i+1}: {metrics.accuracy_score(y_val_folds, y_pred)}")



    print("-"*60)
    print("Fold: %d (%s/%s)" %(i, X_train_fold.shape, X_test_fold.shape))
    print(metrics.classification_report(y_test_fold, preds, target_names=["nonTOR", "TOR"]))
    print("Confusion Matrix: \n%s\n"%metrics.confusion_matrix(y_test_fold, preds))
    print("Log loss : %f" % (metrics.log_loss(y_test_fold, probas)))
    print("AUC      : %f" % (metrics.roc_auc_score(y_test_fold, probas)))
    print("Accuracy : %f" % (metrics.accuracy_score(y_test_fold, preds)))
    print("Precision: %f" % (metrics.precision_score(y_test_fold, preds)))
    print("Recall   : %f" % (metrics.recall_score(y_test_fold, preds)))
    print("F1-score : %f" % (metrics.f1_score(y_test_fold, preds)))

# Log loss : 0.030028
# AUC      : 0.998936
# Accuracy : 0.988933
# Precision: 0.967700
# Recall   : 0.968117
# F1-score : 0.967909
#%%

model.fit(X_train, y_train)
probas = model.predict_proba(X_holdout)[:,1]
preds = (probas > 0.5).astype(int)

print(metrics.classification_report(y_holdout, preds, target_names=["nonTOR", "TOR"]))
print("Confusion Matrix: \n%s\n"%metrics.confusion_matrix(y_holdout, preds))
print("Log loss : %f" % (metrics.log_loss(y_holdout, probas)))
print("AUC      : %f" % (metrics.roc_auc_score(y_holdout, probas)))
print("Accuracy : %f" % (metrics.accuracy_score(y_holdout, preds)))
print("Precision: %f" % (metrics.precision_score(y_holdout, preds)))
print("Recall   : %f" % (metrics.recall_score(y_holdout, preds)))
print("F1-score : %f" % (metrics.f1_score(y_holdout, preds)))

# precision    recall  f1-score   support

#       nonTOR       1.00      0.99      0.99     13930
#          TOR       0.97      0.98      0.97      2901

#     accuracy                           0.99     16831
#    macro avg       0.98      0.99      0.98     16831
# weighted avg       0.99      0.99      0.99     16831

# Confusion Matrix: 
# [[13842    88]
#  [   67  2834]]

# Log loss : 0.025992
# AUC      : 0.999092
# Accuracy : 0.990791
# Precision: 0.969884
# Recall   : 0.976905
# F1-score : 0.973381

# %%


import kmapper as km
import pandas as pd
import numpy as np
from sklearn import ensemble, cluster

df = load_and_prep_data()

features = [
       ' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
       'Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
       'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
       ' Idle Std', ' Idle Max', ' Idle Min',]

X = np.array(df[features])
y = np.array(df.label)

projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(X)
lens1 = projector.decision_function(X)

mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="knn_distance_5")

lens = np.c_[lens1, lens2]



# %%

# G_original = mapper.map(
#     lens,
#     X,
#     cover = km.Cover(n_cubes=20,
#                      perc_overlap=.15),
#     clusterer=cluster.AgglomerativeClustering(3))

from km import KeplerMapper as MyKeplerMapper
lens_functions = [
    projector.decision_function, 
    lambda x:  mapper.fit_transform(x, projection="knn_distance_5")
    ]
my_km = MyKeplerMapper()
lens = my_km.fit_transform(X, lens_functions=lens_functions)
# Using simple resolution and overlap
_ = my_km.map(lens, X, 
        # clusterer=sklearn_cluster.KMeans(
        #     n_clusters=5, # FIXME : still a bit arbitrary
        #     random_state=0
        #     ),
        resolution=20, 
        overlap_ratio=0.15
)

# Convert to kmapper-compatible format
G = my_km.to_kmapper_json()

print(f"num nodes: {len(G['nodes'])}")
print(f"num edges: {sum([len(values) for key, values in G['links'].items()])}")


# %%
html = mapper.visualize(
    G,
    custom_tooltips=y,
    color_values=y,
    color_function_name="target",
    path_html="output/tor-tda.html",
    X=X,
    X_names=list(df[features].columns),
    lens=lens,
    lens_names=["IsolationForest", "KNN-distance 5"],
    title="Detecting encrypted Tor Traffic with Isolation Forest and Nearest Neighbor Distance"
)
# %%
# from kmapper import jupyter
# from IPython.display import Image, display
# jupyter.display("output/tor-tda.html")

with open("output/tor-tda.html", "w") as f:
    f.write(html)
# %%

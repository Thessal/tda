from data import X, y, features
import kmapper as km
import pandas as pd
import numpy as np
from sklearn import ensemble, cluster



projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(X)
lens1 = projector.decision_function(X)

mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="knn_distance_5")

lens = np.c_[lens1, lens2]

# %%

G = mapper.map(
    lens,
    X,
    cover = km.Cover(n_cubes=20,
                     perc_overlap=.15),
    clusterer=cluster.AgglomerativeClustering(3))

print(f"num nodes: {len(G['nodes'])}")
print(f"num edges: {sum([len(values) for key, values in G['links'].items()])}")




# %%
_ = mapper.visualize(
    G,
    custom_tooltips=y,
    color_values=y,
    color_function_name="target",
    path_html="output/fin-tda.html",
    X=X,
    X_names=features,
    lens=lens,
    lens_names=["IsolationForest", "KNN-distance 5"],
    title="Detecting institutional investor from transaction pattern"
)
# %%
# from kmapper import jupyter
# from IPython.display import Image, display
# jupyter.display("output/tor-tda.html")

# with open("output/fin-tda.html", "w") as f:
#     f.write(html)
# %%


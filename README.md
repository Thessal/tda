# tda

## Tor data 

* download `SelectedFeatures-10s-TOR-NonTOR.csv` dataset from https://www.unb.ca/cic/datasets/tor.html
* overall code was taken from : https://kepler-mapper.scikit-tda.org/en/latest/notebooks/TOR-XGB-TDA.html (HJ van Veen)
* main code is tor.py 
* keplermapper implementation is in km.py
* output is a kind of hypergraph. nodes are clusters of covers. node color means the ratio or tor traffic. 

### Questions 

* if a cover is empty (e.g. 3 records/ 1 hypercube when kmeans.n_cluster=5, 3 < 5 == True) then I just assigned all records to the same cluster, which is not logical. 
* Assuming uniform prior looks not very logical. Maybe we need kernel density estimation & transform it into quantile, or some other method. Maybe noninformative prior and bayesian update can be helpful? 
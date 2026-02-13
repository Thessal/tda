import numpy as np
import networkx as nx
from sklearn import cluster as sklearn_cluster
from sklearn import preprocessing
from collections import defaultdict
import itertools

from data import X, y

class KeplerMapper:
    """
    A class for the Mapper algorithm, which builds a topological graph representation 
    of high-dimensional data.
    """
    def __init__(self, verbose=1):
        self.verbose = verbose
        # The final graph will be stored here
        self.graph = nx.Graph() 
        
    def fit_transform(self, X, lens_functions=[lambda x: np.sum(x, axis=1)], scaler=preprocessing.MinMaxScaler()):
        """
        Step 1: Projection (Lens)
        
        This method should project the high-dimensional data into a lower-dimensional space
        (usually 1D or 2D) called the 'lens'.
        
        Args:
            X (numpy.ndarray): The original data. [n_samples, p_features]
            projection (str, callable, or list): How to project the data. 
                                        e.g., "sum", "mean", a function, or a list of these.
            scaler (scaler object): A scaler to normalize the projected data (usually to [0,1]).
        
        Returns:
            lens (numpy.ndarray): The projected and scaled data.
        """

        # A. Apply the projection(s)
        lens_parts = []
        for p in lens_functions:
            part = p(X)
            # Ensure it's column-shaped
            if part.ndim == 1:
                part = part.reshape(-1, 1)
            lens_parts.append(part)
            projected_X = np.hstack(lens_parts) 
            
        # B. Scale the projection
        # The Mapper algorithm usually works best when the lens is scaled to [0, 1]
        lens = scaler.fit_transform(projected_X)
        
        return lens # [n_samples, len(lens_functions)]

    def map(
        self, lens, X, 
        # clusterer= sklearn_cluster.DBSCAN(
        #     eps=0.2, # FIXME : too arbitrary and sensitive! robustness issue
        #     min_samples=3
        #     ),
        clusterer=sklearn_cluster.KMeans(
            n_clusters=5, # FIXME : still a bit arbitrary
            random_state=0
            ),
        resolution=10, 
        overlap_ratio=0.1):
        """
        Step 2, 3, 4: Cover, Cluster, and Graph
        
        This is the core of the Mapper algorithm.
         
        NOTE: It assumes uniform distribution prior. (uniform cover)
        To make this assumption valid, inverse cdf have to be applied to the lens.
        Or, Other noninformative prior should be used and the overlap should be calulated in probability space.

        Args:
            lens (numpy.ndarray): The projected data (output of fit_transform).
            X (numpy.ndarray): The original high-dimensional data.
            clusterer (sklearn estimator): The clustering algorithm (e.g., DBSCAN).
            resolution (int): Number of intervals (hypercubes) to cover the lens with.
            overlap_raio (float): Percentage of overlap between adjacent intervals (0.0 to 1.0). ratio per dimension (unit is (length / length) )
            
        Returns:
            graph (networkx.Graph): A networkx graph where nodes are clusters and edges 
                                    represent overlap between clusters.
        """
        
        # Dictionary to store which points belong to which node (cluster_id -> list of indices)
        nodes = defaultdict(list)
        
        # -------------------------------------------------------------------------
        # Step A: Define the Cover (Hypercubes)
        # -------------------------------------------------------------------------
        
        # 1. Calculate the length (L) of intervals for each dimension
        # Mathematical hint: L = 1 / (resolution - (resolution-1) * overlap_ratio) 
        L = 1 / (resolution - 1) / (1 - overlap_ratio)
        
        # 2. Create intervals for EACH dimension
        # If lens is 1D (vector), make it 2D (matrix with 1 column) for consistency
        if lens.ndim == 1:
            lens = lens.reshape(-1, 1)
            
        n_features = lens.shape[1]
        
        # List of lists: [[(start, end), ... for dim 0], [(start, end), ... for dim 1], ...]
        all_intervals = []
        
        for i in range(n_features):
            starts = np.linspace(0, 1-L, resolution)
            ends = starts + L
            intervals = list(zip(starts, ends))
            all_intervals.append(intervals)
            
        # 3. Create Hypercubes (Cartesian Product)
        # ----------------------------------------------------
        # [TODO] Use itertools.product to create all combinations of intervals
        # hypercubes will be a list of tuples, where each tuple contains 'n_features' intervals
        # e.g. [ ((s1, e1), (s2, e2)), ((s1, e1), (s3, e3)), ... ]
        # ----------------------------------------------------
        hypercubes = list(itertools.product(*all_intervals))

        # -------------------------------------------------------------------------
        # Step B: Loop through hypercubes and Cluster
        # -------------------------------------------------------------------------
        
        for hypercube_idx, hypercube in enumerate(hypercubes):
            
            # hypercube is a tuple of intervals, e.g. ((0.0, 0.2), (0.1, 0.3)) if 2D
            
            # 1. Find indices of points inside this hypercube
            # ----------------------------------------------------
            # [TODO] Create a mask that is True only if point is inside ALL dimension intervals
            # ----------------------------------------------------
            # Start with all True
            mask = np.ones(len(X), dtype=bool)
            
            for i, (start, end) in enumerate(hypercube):
                # Update mask: AND logic for each dimension
                mask = mask & (lens[:, i] >= start) & (lens[:, i] <= end)
                
            data_indices = np.where(mask)[0]
            
            if len(data_indices) == 0:
                continue
            
            # 2. Extract original data for these points
            subset_X = X[data_indices]
            
            # 3. Cluster this subset of data ('subset' : subset[0]:={x | intervals[0].between(x)})
            # ----------------------------------------------------
            # Fit the 'clusterer' on 'subset_X'
            # ----------------------------------------------------
            n_samples = len(subset_X)
            if n_samples < clusterer.n_clusters:
                labels = np.zeros(n_samples, dtype=int)
            else:
                clusterer.fit(subset_X) 
                labels = clusterer.labels_
            
            # 4. Create nodes for each unique cluster found (ignore noise -1 if using DBSCAN)
            unique_labels = np.unique(labels)
            
            for label in unique_labels: # NOTE: cluster index becomes the nodes for each interval
                if label == -1: 
                    print("cluster index is -1. ignoring")
                    continue # Skip noise if there's any
                
                # Get the actual original indices for points in this specific cluster
                cluster_members = data_indices[labels == label]
                
                # Create a unique ID for this node (kmapper convention: "cube_index_cluster_label")
                node_id = f"node_{hypercube_idx}_{label}"
                
                # Store the members
                nodes[node_id] = cluster_members
                
                # Add node to graph with metadata (optional: size, average value, etc.)
                self.graph.add_node(node_id, size=len(cluster_members), members=cluster_members)

        # -------------------------------------------------------------------------
        # Step C: Build Edges (Intersections)
        # -------------------------------------------------------------------------
        # An edge exists if two nodes share at least one data point.
        
        node_ids = list(nodes.keys())
        
        # [TODO] Iterate through all pairs of nodes to find intersections
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u = node_ids[i]
                v = node_ids[j]
                
                members_u = set(nodes[u])
                members_v = set(nodes[v])
                
                # ----------------------------------------------------
                # [TODO] Calculate intersection of members
                # FIXME : Need to check. a bit confused
                # ----------------------------------------------------
                intersection = members_u.intersection(members_v) 
                
                if len(intersection) > 0:
                    # Add edge with weight = size of intersection
                    self.graph.add_edge(u, v, weight=len(intersection))
        
        # Store metadata in the graph
        self.graph.graph["resolution"] = resolution
        self.graph.graph["overlap_ratio"] = overlap_ratio
        self.graph.graph["clusterer"] = str(clusterer)

        return self.graph

    def to_kmapper_json(self):
        """
        Convert the local networkx graph to the dictionary format expected by 
        kmapper.visualize().
        """
        G = self.graph
        
        # 1. Nodes
        nodes = {}
        for node_id, data in G.nodes(data=True):
            # kmapper expects members to be a list of indices
            # We stored them as 'members' in map()
            if "members" in data:
                nodes[node_id] = list(data["members"])
            else:
                nodes[node_id] = []

        # 2. Links (Edges)
        links = defaultdict(list)
        for u, v, data in G.edges(data=True):
            # kmapper expects links to be a dict where key is node_id and value is list of connected node_ids
            # BUT wait, looking at kmapper source/examples, 'links' is a dict of lists?
            # actually, standard kmapper output for 'links' is {node_id: [target_node_id, ...]}
            # checking typical kmapper output...
            links[u].append(v)
            links[v].append(u)
        
        # 3. Simplices (Nodes)
        # In kmapper, simplices usually refers to the nodes themselves if it's a 1-skeleton
        simplices = list(nodes.keys())

        # 4. Meta Data
        meta_data = {
            "projection": "custom", # placeholder
            "n_cubes": G.graph.get("resolution", 10),
            "perc_overlap": G.graph.get("overlap_ratio", 0.1),
            "clusterer": G.graph.get("clusterer", "custom"),
            "scaler": "MinMaxScaler", # placeholder, we used MinMaxScaler by default
            "nerve_min_intersection": 1
        }

        # 5. Meta Nodes (Statistics)
        # This is optional but good to have. We can just leave it empty or minimal
        meta_nodes = {} 

        return {
            "nodes": nodes,
            "links": dict(links),
            "simplices": simplices,
            "meta_data": meta_data,
            "meta_nodes": meta_nodes
        }

    def visualize(self):
        """
        Simple text-based visualization of the graph stats.
        """
        print(f"Graph Created!")
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print(f"Connected Components: {nx.number_connected_components(self.graph)}")

if __name__ == "__main__":
    # Test stub
    # 1. Create a dummy dataset (e.g., a noisy circle)
    print("Generating test data (Circle)...")
    t = np.linspace(0, 2*np.pi, 1000)
    X = np.c_[np.cos(t), np.sin(t)] + np.random.normal(0, 0.1, (1000, 2))
    
    # 2. Init Mapper
    mapper = KeplerMapper()
    
    # 3. Project
    print("Projecting data...")
    sum_lens = lambda x: np.sum(x, axis=1)
    std_lens = lambda x: np.std(x, axis=1)
    lens = mapper.fit_transform(X, lens_functions=[sum_lens, std_lens])
    
    # 4. Map
    print("Mapping...")
    # Using simple resolution and overlap
    graph = mapper.map(lens, X, resolution=5, overlap_ratio=0.2)
    
    # 5. Visualize stats
    mapper.visualize()

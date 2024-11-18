import streamlit as st
import numpy as np
import faiss
from typing import Dict
import hdbscan
from sklearn.cluster import KMeans, DBSCAN, OPTICS

def setup_clustering_configuration(files_data: Dict, embedding_results: Dict) -> Dict:
    """Setup clustering configuration"""
    # Get the correct embedding column
    if "embedding_column" in embedding_results:
        column = embedding_results["embedding_column"]
    elif "embedding_columns" in embedding_results and embedding_results["embedding_columns"]:
        column = embedding_results["embedding_columns"][0]
    else:
        st.error("No embedding columns found")
        return None
    
    config = {
        "type": "clustering",
        "column": column
    }
    
    # Select clustering algorithm
    algorithm = st.sidebar.selectbox(
        "Clustering Algorithm",
        options=["K-Means", "DBSCAN", "HDBSCAN", "OPTICS"],
        help="""
        K-Means: Clusters data into k groups based on centroids
        DBSCAN: Density-based clustering that finds core samples
        HDBSCAN: Hierarchical density-based clustering
        OPTICS: Ordering points to identify clustering structure
        """
    )
    
    config["algorithm"] = algorithm
    
    # Algorithm-specific parameters
    if algorithm == "K-Means":
        n_clusters = st.sidebar.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=100,
            value=5,
            help="Number of clusters to create"
        )
        config["n_clusters"] = n_clusters
        
    elif algorithm == "HDBSCAN":
        min_cluster_size = st.sidebar.number_input(
            "Minimum Cluster Size",
            min_value=2,
            max_value=100,
            value=2,
            help="Minimum number of points required to form a cluster"
        )
        min_samples = st.sidebar.number_input(
            "Minimum Samples",
            min_value=1,
            max_value=100,
            value=1,
            help="Number of samples in a neighborhood for a point to be considered as a core point"
        )
        config.update({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples
        })
        
    elif algorithm in ["DBSCAN", "OPTICS"]:
        min_samples = st.sidebar.number_input(
            "Minimum Points per Cluster",
            min_value=2,
            max_value=100,
            value=2,
            help="Minimum number of points required to form a cluster"
        )
        config["min_samples"] = min_samples
        
        if algorithm == "DBSCAN":
            eps = st.sidebar.number_input(
                "Epsilon (ε)",
                min_value=0.01,
                max_value=2.0,
                value=0.5,
                step=0.01,
                help="Maximum distance between two samples for them to be considered neighbors"
            )
            config["eps"] = eps
        
        # Add max_eps for OPTICS
        elif algorithm == "OPTICS":
            max_eps = st.sidebar.number_input(
                "Maximum Epsilon",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Maximum distance to look for neighbors"
            )
            config["max_eps"] = max_eps
    
    return config

def setup_similarity_configuration(files_data: Dict, embedding_results: Dict) -> Dict:
    """Setup similarity calculation configuration"""
    st.sidebar.header("3. Similarity Calculation")
    
    similarity_method = st.sidebar.radio(
        "Similarity Method",
        options=["Cosine Similarity", "FAISS Search"],
        help="Choose the method to calculate similarity"
    )
    
    # Initialize config with correct keys
    config = {
        "type": "similarity",
        "similarity_method": "cosine" if similarity_method == "Cosine Similarity" else "faiss"
    }
    
    if similarity_method == "FAISS Search":
        num_neighbors = st.sidebar.number_input(
            "Number of nearest neighbors (k)",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of nearest neighbors to find"
        )
        config["num_neighbors"] = num_neighbors  # Changed from "k" to "num_neighbors"
    
    # Add comparison mode configuration
    config.update(setup_comparison_mode(embedding_results))
    
    return config

def calculate_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray, config: Dict) -> Dict:
    """Calculate similarity based on configuration"""
    
    similarity_method = config.get("similarity_method", "faiss")
    
    if similarity_method == "cosine":
        st.write("Using Cosine Similarity")
        similarity_matrix = calculate_cosine_similarity(embeddings1, embeddings2)
        return {"similarity_matrix": similarity_matrix}
        
    else:  # faiss
        st.write("Using FAISS Search")
        k = int(config.get("num_neighbors", 3))
        # Increase k by 1 to account for self-match that we'll remove
        search_k = k + 1
        st.write(f"Using k={k}")
        
        # Initialize FAISS index
        index = faiss.IndexFlatIP(embeddings2.shape[1])
        index.add(embeddings2.astype(np.float32))
        
        # Search for nearest neighbors
        scores, indices = index.search(embeddings1.astype(np.float32), search_k)
        
        # Remove self-matches
        filtered_indices = []
        filtered_scores = []
        
        for i, (idx_row, score_row) in enumerate(zip(indices, scores)):
            # Filter out the self-match and take only k neighbors
            mask = idx_row != i
            filtered_idx = idx_row[mask][:k]
            filtered_score = score_row[mask][:k]
            
            # Pad with -1 if we don't have enough neighbors after filtering
            if len(filtered_idx) < k:
                filtered_idx = np.pad(filtered_idx, (0, k - len(filtered_idx)), constant_values=-1)
                filtered_score = np.pad(filtered_score, (0, k - len(filtered_score)), constant_values=0)
            
            filtered_indices.append(filtered_idx)
            filtered_scores.append(filtered_score)
        
        return {
            "indices": np.array(filtered_indices),
            "similarity_scores": np.array(filtered_scores)
        }

def setup_comparison_mode(embedding_results: Dict) -> Dict:
    """Setup comparison mode configuration"""
    if embedding_results["mode"] == "single":
        return setup_single_mode_comparison(embedding_results)
    else:
        return setup_dual_mode_comparison(embedding_results)

def calculate_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between two sets of embeddings"""
    try:
        # Handle empty or invalid embeddings
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([])

        # Calculate dot product
        dot_product = np.dot(embeddings1, embeddings2.T)
        
        # Calculate magnitudes
        magnitude1 = np.sqrt(np.sum(embeddings1**2, axis=1))
        magnitude2 = np.sqrt(np.sum(embeddings2**2, axis=1))
        
        # Avoid division by zero
        magnitude_matrix = np.outer(magnitude1, magnitude2)
        eps = 1e-8
        
        # Calculate cosine similarity
        similarity_matrix = dot_product / (magnitude_matrix + eps)
        
        # Clip values to [-1, 1] range
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        return similarity_matrix
        
    except Exception as e:
        st.error(f"Error in cosine similarity calculation: {str(e)}")
        return np.array([])

def calculate_faiss_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """Calculate nearest neighbors using FAISS"""
    try:
        # Handle empty or invalid embeddings
        if embeddings1.size == 0 or embeddings2.size == 0:
            return {"indices": np.array([]), "similarity_scores": np.array([])}

        # Convert to float32 as required by FAISS
        embeddings1 = embeddings1.astype('float32')
        embeddings2 = embeddings2.astype('float32')
        
        # Get embedding dimension
        dimension = embeddings2.shape[1]
        
        # Initialize FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings2)
        
        # Adjust k if it's larger than the number of embeddings
        k = min(k, embeddings2.shape[0])
        
        # Search for nearest neighbors
        distances, indices = faiss_index.search(embeddings1, k)
        
        # Convert L2 distances to similarity scores (inverse relationship)
        max_distance = np.max(distances) + 1e-8  # Add small epsilon to avoid division by zero
        similarity_scores = 1 - (distances / max_distance)
        
        return {
            "indices": indices,
            "similarity_scores": similarity_scores
        }
    except Exception as e:
        st.error(f"Error in FAISS similarity calculation: {str(e)}")
        return {"indices": np.array([]), "similarity_scores": np.array([])}

def setup_single_mode_comparison(embedding_results: Dict) -> Dict:
    """Setup comparison configuration for single file mode"""
    comparison_mode = st.sidebar.radio(
        "Comparison Mode",
        options=[
            "Within Single Column",
            "Between Two Columns"
        ] if len(embedding_results["embedding_columns"]) > 1 else ["Within Single Column"]
    )
    
    if comparison_mode == "Within Single Column":
        column = st.sidebar.selectbox(
            "Select Column",
            options=embedding_results["embedding_columns"]
        )
        return {
            "mode": "single_column",
            "column": column
        }
    else:
        col1 = st.sidebar.selectbox(
            "First Column",
            options=embedding_results["embedding_columns"]
        )
        col2 = st.sidebar.selectbox(
            "Second Column",
            options=[c for c in embedding_results["embedding_columns"] if c != col1]
        )
        return {
            "mode": "two_columns",
            "column1": col1,
            "column2": col2
        }

def setup_dual_mode_comparison(embedding_results: Dict) -> Dict:
    """Setup comparison configuration for dual file mode"""
    return {
        "mode": "dual_files",
        "column1": st.sidebar.selectbox(
            "First File Column",
            options=embedding_results["embedding_columns_1"]
        ),
        "column2": st.sidebar.selectbox(
            "Second File Column",
            options=embedding_results["embedding_columns_2"]
        )
    }

def perform_clustering(embeddings: np.ndarray, config: Dict) -> Dict:
    """Perform clustering based on configuration"""
    try:
        if config["algorithm"] == "K-Means":
            kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42)
            labels = kmeans.fit_predict(embeddings)
            # Calculate probabilities based on distance to cluster centers
            distances = kmeans.transform(embeddings)
            probabilities = 1 / (1 + distances.min(axis=1))
            
            return {
                "labels": labels,
                "probabilities": probabilities,
                "inertia": kmeans.inertia_
            }
            
        elif config["algorithm"] == "DBSCAN":
            dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
            labels = dbscan.fit_predict(embeddings)
            
            return {
                "labels": labels
            }
            
        elif config["algorithm"] == "HDBSCAN":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=config["min_cluster_size"],
                min_samples=config["min_samples"],
                prediction_data=True
            )
            labels = clusterer.fit_predict(embeddings)
            
            return {
                "labels": labels,
                "probabilities": clusterer.probabilities_,
                "outlier_scores": clusterer.outlier_scores_
            }
            
        elif config["algorithm"] == "OPTICS":
            optics = OPTICS(
                min_samples=config["min_samples"],
                max_eps=config["max_eps"]
            )
            labels = optics.fit_predict(embeddings)
            outlier_scores = optics.reachability_
            
            return {
                "labels": labels,
                "outlier_scores": outlier_scores
            }
            
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        st.error(f"Config: {config}")
        return None
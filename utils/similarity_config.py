import streamlit as st
import numpy as np
import faiss
from typing import Dict

def setup_clustering_configuration(files_data: Dict, embedding_results: Dict) -> Dict:
    """Setup clustering configuration"""
    st.sidebar.header("3. Clustering Configuration")
    
    # Select column for clustering
    column = st.sidebar.selectbox(
        "Select column for clustering",
        options=embedding_results.get("embedding_columns", []),
        help="Choose which embedded column to use for clustering"
    )
    
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
    
    config = {
        "type": "clustering",
        "column": column,
        "algorithm": algorithm
    }
    
    # Algorithm-specific parameters
    if algorithm == "K-Means":
        config["n_clusters"] = st.sidebar.number_input(
            "Number of clusters (k)",
            min_value=2,
            max_value=100,
            value=5,
            help="The number of clusters to form"
        )
        
    elif algorithm == "DBSCAN":
        config["eps"] = st.sidebar.number_input(
            "Epsilon (neighborhood size)",
            min_value=0.1,
            max_value=10.0,
            value=0.5,
            help="Maximum distance between two samples for neighborhood"
        )
        config["min_samples"] = st.sidebar.number_input(
            "Minimum samples",
            min_value=2,
            max_value=100,
            value=5,
            help="Number of samples in a neighborhood for a core point"
        )
        
    elif algorithm == "HDBSCAN":
        config["min_cluster_size"] = st.sidebar.number_input(
            "Minimum cluster size",
            min_value=2,
            max_value=100,
            value=5,
            help="The minimum size of clusters"
        )
        config["min_samples"] = st.sidebar.number_input(
            "Minimum samples",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of samples in a neighborhood for a core point"
        )
        
    elif algorithm == "OPTICS":
        config["min_samples"] = st.sidebar.number_input(
            "Minimum samples",
            min_value=2,
            max_value=100,
            value=5,
            help="Number of samples in a neighborhood for a core point"
        )
        config["max_eps"] = st.sidebar.number_input(
            "Maximum epsilon",
            min_value=0.1,
            max_value=10.0,
            value=5.0,  # Changed from np.inf to a reasonable default
            help="Maximum distance between two samples for neighborhood"
        )
        config["cluster_method"] = st.sidebar.selectbox(
            "Cluster extraction method",
            options=["xi", "dbscan"],
            help="Method to extract clusters"
        )
        if config["cluster_method"] == "xi":
            config["xi"] = st.sidebar.slider(
                "Xi parameter",
                min_value=0.01,
                max_value=0.99,
                value=0.05,
                help="Determines the minimum steepness on the reachability plot"
            )
        else:  # dbscan
            config["eps"] = st.sidebar.number_input(
                "Epsilon for DBSCAN extraction",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                help="Epsilon parameter for DBSCAN cluster extraction"
            )
    
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
        st.write(f"Using k={k}")
        
        # Initialize FAISS index
        index = faiss.IndexFlatIP(embeddings2.shape[1])
        index.add(embeddings2.astype(np.float32))
        
        # Search for nearest neighbors
        scores, indices = index.search(embeddings1.astype(np.float32), k)
        
        return {
            "indices": indices,
            "similarity_scores": scores
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

        # Normalize the embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1)
        norm2 = np.linalg.norm(embeddings2, axis=1)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        embeddings1_norm = embeddings1 / (norm1[:, np.newaxis] + eps)
        embeddings2_norm = embeddings2 / (norm2[:, np.newaxis] + eps)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        # Clip values to [-1, 1] range to handle numerical errors
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
    """Perform clustering on embeddings"""
    try:
        if config["algorithm"] == "K-Means":
            from sklearn.cluster import KMeans
            clustering = KMeans(n_clusters=config["n_clusters"], random_state=42)
            labels = clustering.fit_predict(embeddings)
            return {
                "labels": labels,
                "algorithm": config["algorithm"],
                "inertia": clustering.inertia_  # Add inertia for K-Means
            }
            
        elif config["algorithm"] == "DBSCAN":
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(
                eps=config["eps"],
                min_samples=config["min_samples"]
            )
            labels = clustering.fit_predict(embeddings)
            return {
                "labels": labels,
                "algorithm": config["algorithm"],
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)
            }
            
        elif config["algorithm"] == "HDBSCAN":
            import hdbscan
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=config["min_cluster_size"],
                min_samples=config["min_samples"] if config["min_samples"] else None,
                prediction_data=True
            )
            labels = clustering.fit_predict(embeddings)
            return {
                "labels": labels,
                "algorithm": config["algorithm"],
                "probabilities": clustering.probabilities_,
                "outlier_scores": clustering.outlier_scores_
            }
            
        elif config["algorithm"] == "OPTICS":
            from sklearn.cluster import OPTICS
            clustering = OPTICS(
                min_samples=config["min_samples"],
                max_eps=config["max_eps"]
            )
            labels = clustering.fit_predict(embeddings)
            return {
                "labels": labels,
                "algorithm": config["algorithm"],
                "reachability": clustering.reachability_,
                "ordering": clustering.ordering_
            }
            
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        st.error(f"Config: {config}")
        return None
import streamlit as st
import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.similarity_config import calculate_similarity
from models.embedding_config import get_embeddings_for_similarity, get_embeddings_for_clustering
from utils.similarity_config import perform_clustering
import traceback

def convert_df(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def reduce_dimensions(embeddings: np.ndarray, method: str) -> np.ndarray:
    """Reduce embedding dimensions to 2D for visualization"""
    if method == "UMAP":
        reducer = umap.UMAP(random_state=42)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)
    
    return reducer.fit_transform(embeddings)

def visualize_cosine_similarity(similarity_matrix: np.ndarray) -> None:
    """Visualize cosine similarity matrix as heatmap"""
    # Create a more readable heatmap
    fig = px.imshow(
        similarity_matrix,
        labels=dict(color="Similarity Score"),
        title="Similarity Matrix Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto"  # Adjust aspect ratio
    )
    
    # Update layout for better readability
    fig.update_layout(
        width=800,
        height=800,
        xaxis_title="Document Index",
        yaxis_title="Document Index"
    )
    
    # Add hover text
    fig.update_traces(
        hovertemplate="Row: %{y}<br>Column: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
    )
    
    st.plotly_chart(fig)
    
    if st.button("Export Heatmap"):
        fig.write_html("similarity_heatmap.html")
        with open("similarity_heatmap.html", "rb") as file:
            st.download_button(
                "Download Heatmap HTML",
                file,
                "similarity_heatmap.html"
            )

def visualize_faiss_results(results: Dict[str, Any]) -> None:
    """Visualize FAISS nearest neighbors results"""
    viz_option = st.radio(
        "Visualization Type",
        options=["Bar Chart", "Scatter Plot"]
    )
    
    if viz_option == "Bar Chart":
        visualize_faiss_bar_chart(results)
    else:
        visualize_faiss_scatter(results)

def visualize_faiss_bar_chart(results: Dict[str, Any]) -> None:
    """Create bar chart for FAISS results"""
    num_queries = min(5, len(results["indices"]))
    fig = go.Figure()
    
    for i in range(num_queries):
        fig.add_trace(go.Bar(
            name=f'Query {i}',
            x=[f'Neighbor {j}' for j in range(len(results["indices"][i]))],
            y=results["similarity_scores"][i]
        ))
    
    fig.update_layout(
        title="Top Similarity Scores per Query",
        xaxis_title="Nearest Neighbors",
        yaxis_title="Similarity Score",
        barmode='group'
    )
    
    st.plotly_chart(fig)

def visualize_faiss_scatter(results: Dict[str, Any]) -> None:
    """Create scatter plot of all similarity scores"""
    scores_flat = results["similarity_scores"].flatten()
    fig = px.scatter(
        x=range(len(scores_flat)),
        y=scores_flat,
        title="All Similarity Scores",
        labels={"x": "Index", "y": "Similarity Score"}
    )
    st.plotly_chart(fig)

def preview_combined_columns(df, columns, num_samples=5):
    """Show preview of how columns will be combined"""
    st.subheader("Preview of Combined Columns")
    
    # Sample a few rows
    sample_df = df.sample(min(num_samples, len(df)))
    
    # Show original columns
    for col in columns:
        st.write(f"**{col}:**")
        st.write(sample_df[col].tolist())
    
    # Show combined text
    st.write("**Combined Text:**")
    combined = sample_df[columns].apply(
        lambda row: ' '.join(row.values.astype(str)), 
        axis=1
    )
    st.write(combined.tolist())

def visualize_clustering_results(df: pd.DataFrame, embeddings: np.ndarray, 
                               clustering_results: Dict, config: Dict) -> None:
    """Visualize clustering results with UMAP and statistics"""
    
    # Create export DataFrame for clusters
    source_column = config["column"].replace("embeddings_", "")
    clusters_df = pd.DataFrame({
        source_column: df[source_column],
        "Cluster": [f"Noise" if label == -1 else f"Cluster {label}" 
                   for label in clustering_results["labels"]]
    })
    
    # Export options
    with st.expander("💾 Export Options", expanded=True):
        tab1, tab2 = st.tabs(["Export Clusters", "Attach to DataFrame"])
        
        with tab1:
            st.write("Preview of clusters:")
            st.dataframe(clusters_df.head(10))
            csv_clusters = convert_df(clusters_df)
            st.download_button(
                label="Download Clusters CSV",
                data=csv_clusters,
                file_name="clusters.csv",
                mime="text/csv",
            )
            
        with tab2:
            st.write("Preview of full DataFrame with clusters:")
            df_with_clusters = df.copy()
            df_with_clusters["Cluster"] = clusters_df["Cluster"]
            st.dataframe(df_with_clusters.head(10))
            csv_full = convert_df(df_with_clusters)
            st.download_button(
                label="Download Full DataFrame CSV",
                data=csv_full,
                file_name="data_with_clusters.csv",
                mime="text/csv",
            )
    
    # UMAP Visualization
    st.write("### UMAP Visualization of Clusters")
    umap_reducer = umap.UMAP(random_state=42)
    embedding_2d = umap_reducer.fit_transform(embeddings)
    
    fig = px.scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        color=[f"Noise" if label == -1 else f"Cluster {label}" 
               for label in clustering_results["labels"]],
        title="UMAP projection of clusters",
        labels={'x': 'UMAP1', 'y': 'UMAP2'}
    )
    st.plotly_chart(fig)
    
    # Cluster Statistics at the bottom
    st.write("### Cluster Statistics:")
    unique_labels = np.unique(clustering_results["labels"])
    cluster_sizes = pd.Series(clustering_results["labels"]).value_counts().sort_index()
    
    # Calculate noise points
    n_noise = sum(1 for label in clustering_results["labels"] if label == -1)
    n_clustered = len(clustering_results["labels"]) - n_noise
    
    # Display overall statistics
    st.write(f"""
    - Total points: {len(clustering_results['labels'])}
    - Clustered points: {n_clustered} ({n_clustered/len(clustering_results['labels'])*100:.1f}%)
    - Noise points (unclustered): {n_noise} ({n_noise/len(clustering_results['labels'])*100:.1f}%)
    """)
    
    # Create cluster summary statistics
    stats_df = pd.DataFrame({
        "Cluster": [f"Noise" if label == -1 else f"Cluster {label}" for label in unique_labels],
        "Size": cluster_sizes.values,
        "Percentage": [f"{(size/len(clustering_results['labels'])*100):.1f}%" 
                      for size in cluster_sizes.values]
    })
    st.dataframe(stats_df)
    
    # Algorithm-specific metrics
    if config["algorithm"] == "K-Means":
        st.write(f"Inertia: {clustering_results.get('inertia', 'N/A')}")

def visualize_similarity_results(results: Dict[str, Any], config: Dict, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Visualize similarity results"""
    
    # Get source column name
    source_column = config["column"].replace("embeddings_", "")
    
    # Export options at the top
    with st.expander("💾 Export Options", expanded=True):
        tab1, tab2 = st.tabs(["Export Text Pairs", "Attach to DataFrame"])
        
        with tab1:
            # Check if we have cosine similarity or FAISS results
            if "similarity_matrix" in results:
                # Process cosine similarity matrix
                similarity_matrix = results["similarity_matrix"]
                pairs = []
                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix[0])):
                        pairs.append({
                            'Text1': df1[source_column].iloc[i],
                            'Text2': df2[source_column].iloc[j],
                            'Similarity': round(float(similarity_matrix[i][j]), 3)
                        })
                pairs_df = pd.DataFrame(pairs)
                pairs_df = pairs_df.sort_values('Similarity', ascending=False)
                
            else:  # FAISS results
                # Process FAISS results
                pairs = []
                for query_idx, (indices, scores) in enumerate(zip(results["indices"], results["similarity_scores"])):
                    query_text = df1[source_column].iloc[query_idx]
                    for idx, score in zip(indices, scores):
                        if idx >= 0:  # Valid index
                            pairs.append({
                                'Text1': query_text,
                                'Text2': df2[source_column].iloc[idx],
                                'Similarity': round(float(score), 3)
                            })
                pairs_df = pd.DataFrame(pairs)
            
            # Show preview
            st.write("Preview of text pairs:")
            st.dataframe(pairs_df.head(10))
            
            # Download button
            csv_pairs = convert_df(pairs_df)
            st.download_button(
                label="Download Text Pairs CSV",
                data=csv_pairs,
                file_name="similarity_pairs.csv",
                mime="text/csv",
            )
        
        with tab2:
            if "similarity_matrix" in results:
                # For cosine similarity, find most similar text for each row
                similarity_matrix = results["similarity_matrix"]
                enhanced_df = df1.copy()
                most_similar = []
                similarity_scores = []
                
                for i in range(len(similarity_matrix)):
                    similarities = similarity_matrix[i].copy()
                    similarities[i] = -1  # Exclude self-similarity
                    most_similar_idx = np.argmax(similarities)
                    most_similar.append(df2[source_column].iloc[most_similar_idx])
                    similarity_scores.append(round(float(similarities[most_similar_idx]), 3))
                
                enhanced_df['Most_Similar_Text'] = most_similar
                enhanced_df['Similarity_Score'] = similarity_scores
                
            else:  # FAISS results
                # For FAISS, add nearest neighbors as lists
                enhanced_df = df1.copy()
                similar_texts = []
                similarity_scores = []
                
                for indices, scores in zip(results["indices"], results["similarity_scores"]):
                    valid_mask = indices >= 0
                    texts = [df2[source_column].iloc[idx] for idx in indices[valid_mask]]
                    scores_list = [round(float(score), 3) for score in scores[valid_mask]]
                    similar_texts.append(texts)
                    similarity_scores.append(scores_list)
                
                enhanced_df['Similar_Texts'] = similar_texts
                enhanced_df['Similarity_Scores'] = similarity_scores
            
            # Show preview
            st.write("Preview of enhanced DataFrame:")
            st.dataframe(enhanced_df.head(10))
            
            # Download button
            csv_enhanced = convert_df(enhanced_df)
            st.download_button(
                label="Download Enhanced DataFrame CSV",
                data=csv_enhanced,
                file_name="enhanced_data.csv",
                mime="text/csv",
            )
    
    # Visualizations
    if "similarity_matrix" in results:
        # Visualize cosine similarity matrix
        st.write("### Similarity Matrix Heatmap")
        visualize_cosine_similarity(results["similarity_matrix"])
    else:
        # Visualize FAISS results
        st.write("### Nearest Neighbors Distribution")
        visualize_faiss_results(results)
    
    # Show top similar pairs
    st.write("### Most Similar Pairs")
    st.dataframe(pairs_df.head(10))

def export_visualization(fig, filename: str):
    """Export visualization to HTML file"""
    if st.button("Export Visualization"):
        fig.write_html(filename)
        with open(filename, "rb") as file:
            st.download_button(
                "Download Visualization HTML",
                file,
                filename,
                "text/html"
            )

def display_embeddings_dataframe(files_data: Dict, embedding_results: Dict) -> None:
    """Display DataFrame with original columns and their embeddings"""
    st.subheader("Generated Embeddings")
    
    if embedding_results["mode"] == "single":
        # Create DataFrame with original columns and embeddings
        display_df = files_data["main_df"].copy()
        for col in embedding_results["embedding_columns"]:
            display_df[col] = display_df[col].apply(lambda x: f"Vector({len(x)} dimensions)")
        
        st.dataframe(display_df)
        
    else:  # dual mode
        # First file
        st.write("First File Embeddings:")
        display_df1 = files_data["main_df"].copy()
        for col in embedding_results["embedding_columns_1"]:
            display_df1[col] = display_df1[col].apply(lambda x: f"Vector({len(x)} dimensions)")
        st.dataframe(display_df1)
        
        # Second file
        st.write("Second File Embeddings:")
        display_df2 = files_data["second_df"].copy()
        for col in embedding_results["embedding_columns_2"]:
            display_df2[col] = display_df2[col].apply(lambda x: f"Vector({len(x)} dimensions)")
        st.dataframe(display_df2)

def perform_and_visualize_analysis(config: Dict[str, Any]) -> None:
    """Perform and visualize analysis based on configuration"""
    try:
        if not hasattr(st.session_state, 'files_data'):
            st.error("No file data found in session state")
            return
            
        if config["type"] == "similarity":
            # Get embeddings from session state
            embeddings1, embeddings2 = get_embeddings_for_similarity(config)
            
            # Calculate similarity
            similarity_results = calculate_similarity(embeddings1, embeddings2, config)
            
            # Visualize results
            visualize_similarity_results(
                similarity_results,
                config,
                st.session_state.files_data["main_df"],
                st.session_state.files_data.get("second_df", st.session_state.files_data["main_df"])
            )
        else:  # clustering
            embeddings = get_embeddings_for_clustering(st.session_state.embedding_results, config)
            if embeddings is not None:
                clustering_results = perform_clustering(embeddings, config)
                visualize_clustering_results(
                    st.session_state.files_data["main_df"],  # Pass DataFrame
                    embeddings,  # Pass embeddings
                    clustering_results,  # Pass clustering results
                    config  # Pass config
                )
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
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
    """Visualize clustering results using dimensionality reduction"""
    
    # Static Content (Always visible)
    st.subheader("Clustering Results")
    
    # Export options
    with st.expander("Export Options"):
        export_tab1, export_tab2 = st.tabs(["Export Clusters", "Append to DataFrame"])
        
        with export_tab1:
            export_df = pd.DataFrame({
                'Cluster': clustering_results["labels"]
            })
            if "probabilities" in clustering_results:
                export_df['Cluster_Probability'] = clustering_results["probabilities"]
            if "outlier_scores" in clustering_results:
                export_df['Outlier_Score'] = clustering_results["outlier_scores"]
                
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "Download Cluster Results",
                csv_data,
                "cluster_results.csv",
                "text/csv"
            )
            
        with export_tab2:
            enhanced_df = df.copy()
            enhanced_df['Cluster'] = clustering_results["labels"]
            if "probabilities" in clustering_results:
                enhanced_df['Cluster_Probability'] = clustering_results["probabilities"]
            if "outlier_scores" in clustering_results:
                enhanced_df['Outlier_Score'] = clustering_results["outlier_scores"]
            
            st.write("Preview of Enhanced DataFrame:")
            st.dataframe(enhanced_df.head())
            
            csv_data = enhanced_df.to_csv(index=False)
            st.download_button(
                "Download Enhanced DataFrame",
                csv_data,
                "enhanced_dataframe.csv",
                "text/csv"
            )
    
    # Cluster Statistics
    unique_labels = np.unique(clustering_results["labels"])
    cluster_sizes = pd.Series(clustering_results["labels"]).value_counts().sort_index()
    
    st.write("Cluster Statistics:")
    stats_df = pd.DataFrame({
        "Cluster": unique_labels,
        "Size": cluster_sizes.values
    })
    st.dataframe(stats_df)
    
    # Algorithm-specific metrics
    if config["algorithm"] == "K-Means":
        st.write(f"Inertia: {clustering_results.get('inertia', 'N/A')}")
        
    elif config["algorithm"] == "DBSCAN":
        n_noise = sum(1 for label in clustering_results["labels"] if label == -1)
        st.write(f"Number of noise points: {n_noise}")
        
    elif config["algorithm"] == "HDBSCAN":
        if "probabilities" in clustering_results:
            st.write("Clustering Probabilities Distribution:")
            fig_prob = px.histogram(
                x=clustering_results["probabilities"],
                nbins=50,
                title="Distribution of Clustering Probabilities"
            )
            st.plotly_chart(fig_prob)

    # Visualization section
    st.subheader("Clustering Visualization")
    
    # Container for UMAP visualization
    viz_container = st.container()
    
    with viz_container:
        with st.spinner('Computing UMAP projection...'):
            # Compute UMAP embeddings
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create visualization
            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=clustering_results["labels"],
                title=f"Clustering Results ({config['algorithm']}) - UMAP Visualization",
                labels={"color": "Cluster"},
                color_continuous_scale='viridis' if config["algorithm"] == "K-Means" else 'turbo'
            )
            
            # Add hover information
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Cluster: %{marker.color}",
                    "x: %{x:.2f}",
                    "y: %{y:.2f}"
                ])
            )
            
            # Update layout for better visibility
            fig.update_layout(
                height=600,
                width=800,
                showlegend=True,
                legend_title_text="Clusters"
            )
            
            st.plotly_chart(fig)

def visualize_similarity_results(results: Dict[str, Any], config: Dict, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Visualize similarity results"""
    
    # Check if we have a similarity matrix (cosine) or indices/scores (FAISS)
    is_cosine = "similarity_matrix" in results
    
    if is_cosine:
        # First show the heatmap
        st.write("Similarity Matrix Heatmap")
        visualize_cosine_similarity(results["similarity_matrix"])
        
        # Get the original text column
        text_column = config["column"].replace("embeddings_", "")
        
        # Show top similar pairs
        st.subheader("Most Similar Pairs")
        similarity_matrix = results["similarity_matrix"]
        
        # Get indices of top similar pairs (excluding self-similarity)
        n_top = 10  # Number of top pairs to show
        pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[0])):
                pairs.append({
                    'Text 1': df1[text_column].iloc[i],
                    'Text 2': df2[text_column].iloc[j],
                    'Similarity Score': round(float(similarity_matrix[i][j]), 3)
                })
        
        # Convert to DataFrame and sort
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('Similarity Score', ascending=False).head(n_top)
        
        # Display as interactive DataFrame
        st.dataframe(
            pairs_df,
            column_config={
                'Similarity Score': st.column_config.NumberColumn(
                    format="%.3f",
                ),
            },
            hide_index=True
        )

        # Add export options
        with st.expander("Export Data"):
            export_tab1, export_tab2 = st.tabs(["Export Text Pairs", "Append to DataFrame"])
            
            with export_tab1:
                # Export text pairs as new CSV
                pairs_df = pd.DataFrame(pairs)
                pairs_df = pairs_df.sort_values('Similarity Score', ascending=False)
                csv_data = pairs_df.to_csv(index=False)
                
                st.download_button(
                    "Download Similar Text Pairs",
                    csv_data,
                    "similar_text_pairs.csv",
                    "text/csv"
                )
            
            with export_tab2:
                # Option to append to existing DataFrame
                st.write("Append most similar text to your data:")
                
                # Let user select which DataFrame to append to
                if "second_df" in st.session_state.files_data:
                    df_choice = st.radio(
                        "Select DataFrame to append to:",
                        ["First DataFrame", "Second DataFrame"]
                    )
                    target_df = (st.session_state.files_data["main_df"] if df_choice == "First DataFrame" 
                               else st.session_state.files_data["second_df"])
                else:
                    target_df = st.session_state.files_data["main_df"]
                
                # Create new DataFrame with original data plus most similar text
                enhanced_df = target_df.copy()
                
                # Get most similar text for each row
                similarity_dict = {}
                for i in range(len(similarity_matrix)):
                    similarities = similarity_matrix[i]
                    similarities[i] = -1  # Exclude self-similarity
                    most_similar_idx = np.argmax(similarities)
                    similarity_dict[i] = {
                        'Most_Similar_Text': df2[text_column].iloc[most_similar_idx],
                        'Similarity_Score': similarities[most_similar_idx]
                    }
                
                # Add new columns
                enhanced_df['Most_Similar_Text'] = enhanced_df.index.map(
                    lambda x: similarity_dict[x]['Most_Similar_Text'] if x in similarity_dict else None
                )
                enhanced_df['Similarity_Score'] = enhanced_df.index.map(
                    lambda x: similarity_dict[x]['Similarity_Score'] if x in similarity_dict else None
                )
                
                # Preview enhanced DataFrame
                st.write("Preview of Enhanced DataFrame:")
                st.dataframe(enhanced_df.head())
                
                # Export enhanced DataFrame
                csv_data = enhanced_df.to_csv(index=False)
                st.download_button(
                    "Download Enhanced DataFrame",
                    csv_data,
                    "enhanced_dataframe.csv",
                    "text/csv"
                )
    else:  # faiss
        st.subheader("Nearest Neighbors")
        
        # Get the original text column and k value
        text_column = config["column"].replace("embeddings_", "")
        k = int(config.get("num_neighbors", 3))
        
        # Process all queries and their k nearest neighbors
        query_texts = []
        similar_texts = []
        similarity_scores = []
        
        for query_idx, (indices, scores) in enumerate(zip(results["indices"], results["similarity_scores"])):
            query_text = df1[text_column].iloc[query_idx]
            
            # Ensure we get exactly k neighbors
            indices = indices[:k]
            scores = scores[:k]
            
            neighbor_texts = [df2[text_column].iloc[idx] for idx in indices]
            neighbor_scores = [f"{float(score):.3f}" for score in scores]
            
            query_texts.append(query_text)
            similar_texts.append(neighbor_texts)
            similarity_scores.append(neighbor_scores)
        
        # Create DataFrame with lists in columns
        results_df = pd.DataFrame({
            'Query Text': query_texts,
            'Similar Texts': similar_texts,
            'Similarity Scores': similarity_scores
        })
        
        # Display as interactive DataFrame
        st.dataframe(
            results_df,
            column_config={
                'Query Text': st.column_config.TextColumn(
                    width="medium",
                ),
                'Similar Texts': st.column_config.ListColumn(
                    width="large",
                ),
                'Similarity Scores': st.column_config.ListColumn(
                    width="medium",
                ),
            },
            hide_index=True
        )
        
        # Add export options
        with st.expander("Export Data"):
            export_tab1, export_tab2 = st.tabs(["Export Results", "Append to DataFrame"])
            
            with export_tab1:
                # Create a flattened version of the results for export
                flat_results = []
                for query_text, neighbors, scores in zip(query_texts, similar_texts, similarity_scores):
                    for neighbor_text, score in zip(neighbors, scores):
                        flat_results.append({
                            'Query Text': query_text,
                            'Similar Text': neighbor_text,
                            'Similarity Score': float(score)
                        })
                
                flat_df = pd.DataFrame(flat_results)
                
                # Export as CSV
                csv = flat_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "similarity_results.csv",
                    "text/csv",
                    key="download_results"
                )
                
                # Preview of flattened results
                st.write("Preview of export format:")
                st.dataframe(
                    flat_df.head(),
                    column_config={
                        'Similarity Score': st.column_config.NumberColumn(
                            format="%.3f"
                        )
                    },
                    hide_index=True
                )
            
            with export_tab2:
                st.write("Append results to original DataFrame:")
                
                # Let user select which DataFrame to append to
                if "second_df" in st.session_state.files_data:
                    df_choice = st.radio(
                        "Select DataFrame to append to:",
                        ["First DataFrame", "Second DataFrame"]
                    )
                    target_df = (df1 if df_choice == "First DataFrame" else df2)
                else:
                    target_df = df1
                
                # Create enhanced DataFrame by merging with results
                enhanced_df = target_df.copy()
                enhanced_df['Similar_Texts'] = similar_texts
                enhanced_df['Similarity_Scores'] = similarity_scores
                
                # Preview enhanced DataFrame
                st.write("Preview of Enhanced DataFrame:")
                st.dataframe(
                    enhanced_df,
                    column_config={
                        'Similar_Texts': st.column_config.ListColumn(
                            width="large",
                        ),
                        'Similarity_Scores': st.column_config.ListColumn(
                            width="medium",
                        ),
                    }
                )
                
                # Export enhanced DataFrame
                csv = enhanced_df.to_csv(index=False)
                st.download_button(
                    "Download Enhanced DataFrame",
                    csv,
                    "enhanced_dataframe_with_neighbors.csv",
                    "text/csv",
                    key="download_enhanced"
                )

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
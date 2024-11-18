import streamlit as st
import pandas as pd
from models.embedding_config import (
    generate_embeddings_for_config,
    setup_existing_embeddings,
    setup_new_embeddings
)
from utils.similarity_config import (
    setup_similarity_configuration,
    setup_clustering_configuration,
)
from models.model_constants import *
from utils.utils import set_page_config
from utils.validation import validate_embedding_config
from utils.visualization import (
    perform_and_visualize_analysis
)
from utils.file_config import setup_file_configuration



def setup_analysis_configuration(files_data, embedding_results):
    """Setup analysis configuration (similarity or clustering)"""
    st.sidebar.header("3. Analysis Configuration")
    
    # Determine if clustering is possible
    if embedding_results["mode"] == "single":
        # Get embedding columns based on mode
        if "embedding_column" in embedding_results:
            embedding_columns = [embedding_results["embedding_column"]]
        else:
            embedding_columns = embedding_results.get("embedding_columns", [])
        
        can_cluster = (
            len(embedding_columns) == 1 or 
            "Combine Columns" in str(embedding_results.get("embedding_mode", ""))
        )
    else:
        can_cluster = False
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        options=["Similarity Calculation", "Clustering"] if can_cluster else ["Similarity Calculation"],
        help="Choose the type of analysis to perform"
    )
    
    if analysis_type == "Similarity Calculation":
        return setup_similarity_configuration(files_data, embedding_results)
    else:
        # Store analysis type in session state
        st.session_state.analysis_type = "clustering"
        return setup_clustering_configuration(files_data, embedding_results)



def export_results(results, config, df=None):
    """Export results in various formats"""
    st.subheader("Export Options")
    
    # Create columns for different export options
    
        # Export as CSV
    if config["type"] == "clustering":
        if st.button("Export Results as CSV"):
            df_export = df.copy()
            df_export["Cluster"] = results["labels"]
            if "probabilities" in results:
                df_export["Cluster_Probability"] = results["probabilities"]
            
            st.download_button(
                "Download CSV",
                df_export.to_csv(index=False),
                "clustering_results.csv",
                "text/csv"
            )
    else:  # Similarity results
        if st.button("Export Results as CSV"):
            if config["type"] == "Cosine Similarity":
                pd.DataFrame(results).to_csv("similarity_matrix.csv")
            else:  # FAISS
                pd.DataFrame({
                    'Query_Index': range(len(results["indices"])),
                    'Nearest_Neighbors': [list(idx) for idx in results["indices"]],
                    'Similarity_Scores': [list(scores) for scores in results["similarity_scores"]]
                }).to_csv("similarity_results.csv")

def store_embedding_config(files_data, embedding_config, embedding_results=None):
    """Safely store embedding configuration and results"""
    try:
        # Validate configuration
        is_valid, message = validate_embedding_config(embedding_config)
        if not is_valid:
            st.error(message)
            return False
        
        # Store in session state
        st.session_state.update({
            "embedding_results": embedding_results or embedding_config.get("existing_embeddings"),
            "files_data": files_data,  # This now contains the DataFrame with embeddings
            "embedding_config": embedding_config
        })
        
        return True
        
    except Exception as e:
        st.error(f"Error storing configuration: {str(e)}")
        return False

def main():
    set_page_config("Vector Embedding Analysis", "📐")
    st.title("Vector Embedding Analysis")
    with st.expander("Before you begin"):
        # Read README.md and extract content until "## Before Using This App"
        with open("README.md", "r", encoding="utf-8") as file:
            content = []
            for line in file:
                if line.strip() == "## Before Using This App":
                    break
                content.append(line)
        
        # Display the extracted content
        st.markdown("".join(content))
    
    # Initialize session state if not already done
    if 'files_data' not in st.session_state:
        st.session_state.files_data = None
    if 'embedding_results' not in st.session_state:
        st.session_state.embedding_results = None
    if 'embedding_config' not in st.session_state:
        st.session_state.embedding_config = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'upload'
    
    # Step 1: File Configuration
    if st.session_state.files_data is None:
        files_data = setup_file_configuration()
        if files_data:
            st.session_state.files_data = files_data
    
    if st.session_state.files_data:
        # Step 2: Embedding Configuration
        embedding_mode = st.sidebar.radio(
            "Embedding Mode",
            options=["Generate New Embeddings", "Use Existing Embeddings"],
            key="embedding_mode"
        )
        
        if embedding_mode == "Generate New Embeddings":
            embedding_config = setup_new_embeddings(st.session_state.files_data)
            if embedding_config:
                if st.sidebar.button("Generate Embeddings"):
                    try:
                        embedding_results = generate_embeddings_for_config(
                            st.session_state.files_data, 
                            embedding_config
                        )
                        if embedding_results:
                            st.success("Embeddings generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating embeddings: {str(e)}")
        
        else:  # Use Existing Embeddings
            embedding_config = setup_existing_embeddings(st.session_state.files_data)
            if embedding_config:
                st.session_state.embedding_results = embedding_config
                st.session_state.embedding_config = embedding_config
        
        # Step 3: Analysis Configuration
        if st.session_state.embedding_results:
            analysis_config = setup_analysis_configuration(
                st.session_state.files_data,
                st.session_state.embedding_results
            )
            if analysis_config:
                perform_and_visualize_analysis(analysis_config)


if __name__ == "__main__":
    main()
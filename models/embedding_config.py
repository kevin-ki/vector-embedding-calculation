from openai import OpenAI
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import voyageai
from typing import List, Dict, Tuple
import numpy as np
from stqdm import stqdm
import pandas as pd
import streamlit as st
from models.model_constants import *


class EmbeddingModel:
    def __init__(self, model_type: str, model_name: str, api_key: str = None):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAI(api_key=self.api_key)
            
        elif self.model_type == "sentence-transformer":
            return SentenceTransformer(self.model_name)
            
        elif self.model_type == "google":
            if not self.api_key:
                raise ValueError("Google API key is required")
            genai.configure(api_key=self.api_key)
            return genai
            
        elif self.model_type == "voyage":
            if not self.api_key:
                raise ValueError("Voyage AI API key is required")
            return voyageai.Client(api_key=self.api_key)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_embeddings(self, df: pd.DataFrame, source_columns: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for the specified columns with progress bar
        
        Args:
            df: DataFrame containing the source columns
            source_columns: List of column names to use for embedding generation
            batch_size: Number of texts to process at once (for memory efficiency)
        """
        # Combine specified columns into single text
        df['combined_text'] = df[source_columns].apply(
            lambda row: ' '.join(row.values.astype(str)), 
            axis=1
        )
        
        texts = df['combined_text'].tolist()
        embeddings_list = []
        
        # Process in batches with progress bar
        for i in stqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            if self.model_type == "openai":
                batch_embeddings = [
                    self.model.embeddings.create(
                        input=text, 
                        model=self.model_name
                    ).data[0].embedding 
                    for text in batch_texts
                ]
            
            elif self.model_type == "sentence-transformer":
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            
            elif self.model_type == "google":
                response = self.model.embed_content(
                    model=self.model_name,
                    content=batch_texts,
                    task_type="retrieval_document"
                )
                batch_embeddings = response['embeddings']
            
            elif self.model_type == "voyage":
                result = self.model.embed(
                    batch_texts, 
                    model=self.model_name,
                    input_type="document"
                )
                batch_embeddings = result.embeddings
            
            embeddings_list.extend(batch_embeddings)
        
        # Clean up temporary column
        df.drop('combined_text', axis=1, inplace=True)
        
        return np.array(embeddings_list)
    
def setup_model_selection():
    """Setup for model selection when creating new embeddings"""
    provider = st.sidebar.selectbox(
        "Select Model Provider",
        options=["Sentence Transformer", "OpenAI", "Voyage AI", "Google"],
        help="""Choose the AI provider for generating embeddings:
        - Sentence Transformer: Free, open-source models 
        - OpenAI: requires key from https://platform.openai.com/api-keys
        - Voyage AI: requires key from https://dash.voyageai.com/api-keys
        - Google: requires key from https://aistudio.google.com/apikey"""
    )
    
    model_config = {"type": provider.lower().replace(" ", "-")}
    
    # Show appropriate model options based on provider
    if provider == "OpenAI":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(OPENAI_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} - {OPENAI_EMBEDDING_MODELS[x]}"
        )
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key from https://platform.openai.com/api-keys"
        )
        model_config.update({"model": model_name, "api_key": api_key})
    
    elif provider == "Voyage AI":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(VOYAGE_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} - {VOYAGE_EMBEDDING_MODELS[x]}"
        )
        api_key = st.sidebar.text_input(
            "Voyage AI API Key", 
            type="password",
            help="Enter your Voyage AI API key from https://dash.voyageai.com/api-keys"
        )
        model_config.update({"model": model_name, "api_key": api_key})
    
    elif provider == "Sentence Transformer":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(SENTENCE_TRANSFORMER_MODELS.keys()),
            format_func=lambda x: f"{x} - {SENTENCE_TRANSFORMER_MODELS[x]}",
            help="Select from available open-source models. No API key required."
        )
        model_config.update({"model": model_name})
    
    elif provider == "Google":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(GOOGLE_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} - {GOOGLE_EMBEDDING_MODELS[x]}"
        )
        api_key = st.sidebar.text_input(
            "Google API Key", 
            type="password",
            help="Enter your Google AI API key from https://aistudio.google.com/apikey"
        )
        model_config.update({"model": model_name, "api_key": api_key})
    
    return model_config
def setup_single_file_existing_embeddings(files_data: Dict) -> Dict:
    """Setup configuration for existing embeddings in single file mode"""
    embedding_columns = [col for col in files_data["main_df"].columns 
                        if col.startswith('embeddings_')]
    
    if not embedding_columns:
        st.sidebar.error("No embedding columns found in the file")
        return None
    
    selected_columns = st.sidebar.multiselect(
        "Select embedding columns",
        options=embedding_columns,
        help="Choose the columns containing embeddings"
    )
    
    if selected_columns:
        embeddings = {}
        for col in selected_columns:
            try:
                embeddings[col] = np.array([
                    eval(emb) if isinstance(emb, str) else emb 
                    for emb in files_data["main_df"][col]
                ])
            except Exception as e:
                st.sidebar.error(f"Error loading embeddings from column {col}: {str(e)}")
                return None
        
        return {
            "mode": "single",
            "existing_embeddings": {
                "mode": "single",
                "embedding_columns": selected_columns,
                "embeddings": embeddings
            }
        }
    return None

def combine_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Combine multiple text columns into a single series"""
    return df[columns].apply(
        lambda row: ' '.join(row.values.astype(str)), 
        axis=1
    )

def get_embedding_columns(df: pd.DataFrame) -> List[str]:
    """Get all embedding columns from dataframe"""
    return [col for col in df.columns if col.startswith('embeddings_')]

def setup_embedding_configuration(files_data):
    """Setup embedding configuration based on file mode"""
    st.sidebar.header("2. Embedding Configuration")
    
    if not files_data:
        st.sidebar.warning("Please upload file(s) first")
        return None
        
    embedding_source = st.sidebar.radio(
        "Embedding Source",
        options=["Create New Embeddings", "Use Existing Embeddings" ],
        help="Choose whether to use existing embeddings or create new ones"
    )
    
    config = {"source": embedding_source}
    
    if embedding_source == "Use Existing Embeddings":
        existing_config = setup_existing_embeddings(files_data)
        if existing_config is None:
            return None
        config.update(existing_config)
    else:
        new_config = setup_new_embeddings(files_data)
        if new_config is None:
            return None
        config.update(new_config)
    
    return config

def setup_existing_embeddings(files_data: Dict) -> Dict:
    """Setup configuration for using existing embeddings"""
    config = {}
    
    # Get available embedding columns
    main_embedding_columns = get_embedding_columns(files_data["main_df"])
    if not main_embedding_columns:
        st.sidebar.error("No embedding columns found in the main file")
        return None
        
    if files_data["mode"] == "single":
        config.update({
            "mode": "single",
            "embedding_columns": main_embedding_columns,
            "embeddings": {col: files_data["main_df"][col].values for col in main_embedding_columns}
        })
    else:  # dual mode
        second_embedding_columns = get_embedding_columns(files_data["second_df"])
        if not second_embedding_columns:
            st.sidebar.error("No embedding columns found in the second file")
            return None
            
        config.update({
            "mode": "dual",
            "embedding_columns_1": main_embedding_columns,
            "embedding_columns_2": second_embedding_columns,
            "embeddings_1": {col: files_data["main_df"][col].values for col in main_embedding_columns},
            "embeddings_2": {col: files_data["second_df"][col].values for col in second_embedding_columns}
        })
    
    return config

def generate_embeddings_for_config(files_data: Dict, config: Dict) -> Dict:
    """Generate embeddings based on configuration"""
    st.write("Starting embedding generation...")
    
    try:
        if config["mode"] == "single":
            result = generate_single_file_embeddings(files_data["main_df"], config)
        else:
            result = generate_dual_file_embeddings(files_data, config)
        return result
        
    except Exception as e:
        st.error(f"Error during embedding generation: {str(e)}")
        return None

def generate_single_file_embeddings(df: pd.DataFrame, config: Dict) -> Dict:
    """Generate embeddings for a single file"""
    # Create embedding model
    model = EmbeddingModel(
        model_type=config["type"],
        model_name=config["model"],
        api_key=config.get("api_key")
    )
    
    # Generate embeddings
    column_name = f"embeddings_{'_'.join(config['source_columns'])}"
    embeddings = model.generate_embeddings(
        df=df,
        source_columns=config["source_columns"]
    )
    
    # Ensure embeddings are float32 and store in DataFrame
    embeddings = embeddings.astype(np.float32)
    df[column_name] = embeddings.tolist()
    
    return {
        "mode": "single",
        "embedding_columns": [column_name],
        "embeddings": {column_name: embeddings}
    }

def generate_dual_file_embeddings(files_data: Dict, config: Dict) -> Dict:
    """Generate embeddings for two files"""
    model = EmbeddingModel(
        model_type=config["type"],
        model_name=config["model"],
        api_key=config.get("api_key")
    )
    
    progress_text = "Generating embeddings for both files..."
    progress_bar = st.progress(0)
    
    # Generate embeddings for first file
    st.write("Processing first file...")
    column_name_1 = f"embeddings_{'_'.join(config['source_columns_1'])}"
    embeddings_1 = model.generate_embeddings(
        df=files_data["main_df"],
        source_columns=config["source_columns_1"]
    )
    files_data["main_df"][column_name_1] = embeddings_1.tolist()
    progress_bar.progress(50)
    
    # Generate embeddings for second file
    st.write("Processing second file...")
    column_name_2 = f"embeddings_{'_'.join(config['source_columns_2'])}"
    embeddings_2 = model.generate_embeddings(
        df=files_data["second_df"],
        source_columns=config["source_columns_2"]
    )
    files_data["second_df"][column_name_2] = embeddings_2.tolist()
    progress_bar.progress(100)
    
    st.success("Embedding generation complete!")
    
    return {
        "mode": "dual",
        "embedding_columns_1": [column_name_1],
        "embedding_columns_2": [column_name_2],
        "embeddings_1": {column_name_1: embeddings_1},
        "embeddings_2": {column_name_2: embeddings_2}
    }

def setup_single_file_new_embeddings(files_data: Dict) -> Dict:
    """Setup configuration for creating new embeddings from a single file"""
    text_columns = [col for col in files_data["main_df"].columns 
                   if files_data["main_df"][col].dtype == 'object']
    
    if not text_columns:
        st.sidebar.error("No text columns found in the file")
        return None
        
    embedding_mode = st.sidebar.radio(
        "Embedding Mode",
        options=["Single Column", "Combine Columns"],
        help="Choose how to create embeddings from text columns"
    )
    
    config = {
        "mode": "single",
        "embedding_mode": embedding_mode
    }
    
    if embedding_mode == "Single Column":
        source_column = st.sidebar.selectbox(
            "Select Text Column",
            options=text_columns,
            help="Choose the column to generate embeddings from"
        )
        config["source_columns"] = [source_column]
    else:
        source_columns = st.sidebar.multiselect(
            "Select Text Columns to Combine",
            options=text_columns,
            help="Choose columns to combine for embedding generation"
        )
        if not source_columns:
            st.sidebar.error("Please select at least one column")
            return None
        config["source_columns"] = source_columns
    
    # Add model configuration
    model_config = setup_model_selection()
    if not model_config:
        return None
    
    config.update(model_config)
    return config

def setup_new_embeddings(files_data: Dict) -> Dict:
    """Setup configuration for creating new embeddings"""
    # Get model configuration first
    model_config = setup_model_selection()
    if not model_config:
        return None
        
    if files_data["mode"] == "single":
        # Get text columns
        text_columns = [col for col in files_data["main_df"].columns 
                       if files_data["main_df"][col].dtype == 'object']
        
        if not text_columns:
            st.sidebar.error("No text columns found in the file")
            return None
            
        selected_columns = st.sidebar.multiselect(
            "Select text columns for embedding",
            options=text_columns,
            help="Choose the columns to generate embeddings from"
        )
        
        if not selected_columns:
            st.sidebar.error("Please select at least one column")
            return None
            
        return {
            "mode": "single",
            "source_columns": selected_columns,
            **model_config
        }
    else:
        # Get text columns from both files
        main_text_columns = [col for col in files_data["main_df"].columns 
                           if files_data["main_df"][col].dtype == 'object']
        second_text_columns = [col for col in files_data["second_df"].columns 
                             if files_data["second_df"][col].dtype == 'object']
        
        if not main_text_columns or not second_text_columns:
            st.sidebar.error("No text columns found in one or both files")
            return None
            
        main_selected = st.sidebar.multiselect(
            "Select text columns from first file",
            options=main_text_columns,
            help="Choose columns from first file for embedding generation"
        )
        
        second_selected = st.sidebar.multiselect(
            "Select text columns from second file",
            options=second_text_columns,
            help="Choose columns from second file for embedding generation"
        )
        
        if not main_selected or not second_selected:
            st.sidebar.error("Please select columns from both files")
            return None
            
        return {
            "mode": "dual",
            "source_columns_1": main_selected,
            "source_columns_2": second_selected,
            **model_config
        }
    
def setup_dual_file_new_embeddings(files_data: Dict) -> Dict:
    """Setup configuration for creating new embeddings from two files"""
    # Get text columns from both files
    main_text_columns = [col for col in files_data["main_df"].columns 
                        if files_data["main_df"][col].dtype == 'object']
    second_text_columns = [col for col in files_data["second_df"].columns 
                         if files_data["second_df"][col].dtype == 'object']
    
    if not main_text_columns or not second_text_columns:
        st.sidebar.error("No text columns found in one or both files")
        return None
    
    config = {
        "mode": "dual",
        "source_columns_1": st.sidebar.multiselect(
            "Select Text Columns (First File)",
            options=main_text_columns,
            help="Choose columns from first file for embedding generation"
        ),
        "source_columns_2": st.sidebar.multiselect(
            "Select Text Columns (Second File)",
            options=second_text_columns,
            help="Choose columns from second file for embedding generation"
        )
    }
    
    if not config["source_columns_1"] or not config["source_columns_2"]:
        st.sidebar.error("Please select columns from both files")
        return None
    
    # Add model configuration
    model_config = setup_model_selection()
    if not model_config:
        return None
    
    config.update(model_config)
    return config

def get_embeddings_for_similarity(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Get embeddings based on similarity configuration"""
    try:
        embedding_results = st.session_state.embedding_results
        
        if config["mode"] == "single_column":
            # Get embeddings from embedding_results
            embeddings = np.array(embedding_results["embeddings"][config["column"]], dtype=np.float32)
            return embeddings, embeddings
            
        elif config["mode"] == "two_columns":
            # Get both columns from embedding_results
            embeddings1 = np.array(embedding_results["embeddings"][config["column1"]], dtype=np.float32)
            embeddings2 = np.array(embedding_results["embeddings"][config["column2"]], dtype=np.float32)
            return embeddings1, embeddings2
            
        else:  # dual_files mode
            # Get embeddings from both files in embedding_results
            embeddings1 = np.array(embedding_results["embeddings_1"][config["column1"]], dtype=np.float32)
            embeddings2 = np.array(embedding_results["embeddings_2"][config["column2"]], dtype=np.float32)
            return embeddings1, embeddings2
            
    except Exception as e:
        st.error(f"Error accessing embeddings: {str(e)}")
        st.error(f"Available columns in embedding_results: {list(st.session_state.embedding_results.get('embeddings', {}).keys())}")
        raise

def get_embeddings_for_clustering(embedding_results: Dict, config: Dict) -> np.ndarray:
    """Get embeddings for clustering analysis"""
    try:
        # Get the selected embedding column from embedding_results
        column = config["column"]
        
        # Convert embeddings to numpy array
        embeddings = np.array(
            st.session_state.files_data["main_df"][column].tolist(),
            dtype=np.float32
        )
        
        return embeddings
        
    except Exception as e:
        st.error(f"Error getting embeddings for clustering: {str(e)}")
        return None
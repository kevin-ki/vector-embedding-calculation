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
            
        elif self.model_type == "voyageai":
            if not self.api_key:
                raise ValueError("Voyage AI API key is required")
            return voyageai.Client(api_key=self.api_key)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for the given texts"""
        embeddings_list = []
        
        # Process in batches with progress bar
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.model_type == "openai":
                response = self.model.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                
            elif self.model_type == "sentence-transformer":
                batch_embeddings = self.model.encode(batch).tolist()
                
            elif self.model_type == "google":
                model = genai.GenerativeModel(self.model_name)
                batch_embeddings = [
                    model.embed_content(text=text).embedding
                    for text in batch
                ]
                
            elif self.model_type == "voyageai":
                result = self.model.embed(
                    batch,
                    model=self.model_name,
                    input_type="document"
                )
                batch_embeddings = result.embeddings
                
            embeddings_list.extend(batch_embeddings)
        
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
    
    provider_types = {
        "Sentence Transformer": "sentence-transformer",
        "OpenAI": "openai",
        "Voyage AI": "voyageai",
        "Google": "google"
    }
    
    model_config = {"type": provider_types[provider]}
    
    if provider == "Voyage AI":
        # Use exact model names as per documentation
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=["voyage-2", "voyage-3"],
            format_func=lambda x: f"{x} - {VOYAGE_EMBEDDING_MODELS[x]['description']}"
        )
        api_key = st.sidebar.text_input(
            "Voyage AI API Key", 
            type="password",
            help="Enter your Voyage AI API key"
        )
        # Use the model name directly
        model_config.update({
            "model": model_name,  # No need for model_name mapping anymore
            "api_key": api_key
        })
    
    elif provider == "OpenAI":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(OPENAI_EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} - {OPENAI_EMBEDDING_MODELS[x]}"
        )
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        model_config.update({"model": model_name, "api_key": api_key})
    
    elif provider == "Sentence Transformer":
        model_name = st.sidebar.selectbox(
            "Select Model",
            options=list(SENTENCE_TRANSFORMER_MODELS.keys()),
            format_func=lambda x: f"{x} - {SENTENCE_TRANSFORMER_MODELS[x]}"
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
            help="Enter your Google API key"
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
    st.sidebar.header("3. Embeddings")
    
    if files_data["mode"] == "single":
        # Let user select which column contains embeddings
        embedding_column = st.sidebar.selectbox(
            "Select column containing embeddings",
            options=files_data["main_df"].columns,
            help="Choose which column contains your vector embeddings"
        )
        
        # Let user select which column(s) were used to create these embeddings
        source_columns = st.sidebar.multiselect(
            "Select source column(s)",
            options=files_data["main_df"].columns,
            help="Choose the column(s) that were used to generate these embeddings"
        )
        
        if not source_columns:
            st.sidebar.error("Please select at least one source column")
            return None
            
        try:
            # Convert embeddings to numpy array
            embeddings = np.array([
                eval(emb) if isinstance(emb, str) else emb 
                for emb in files_data["main_df"][embedding_column]
            ], dtype=np.float32)
            
            return {
                "mode": "single",
                "type": "existing",
                "embedding_column": embedding_column,
                "embedding_columns": [embedding_column],
                "source_columns": source_columns,
                "embeddings": embeddings
            }
        except Exception as e:
            st.sidebar.error(f"Error loading embeddings: {str(e)}")
            return None
            
    else:  # dual mode
        # First file
        embedding_column_1 = st.sidebar.selectbox(
            "Select embedding column from first file",
            options=files_data["main_df"].columns,
            help="Choose which column contains embeddings in the first file"
        )
        
        source_columns_1 = st.sidebar.multiselect(
            "Select source column(s) from first file",
            options=files_data["main_df"].columns,
            help="Choose the column(s) that were used to generate these embeddings"
        )
        
        # Second file
        embedding_column_2 = st.sidebar.selectbox(
            "Select embedding column from second file",
            options=files_data["second_df"].columns,
            help="Choose which column contains embeddings in the second file"
        )
        
        source_columns_2 = st.sidebar.multiselect(
            "Select source column(s) from second file",
            options=files_data["second_df"].columns,
            help="Choose the column(s) that were used to generate these embeddings"
        )
        
        if not source_columns_1 or not source_columns_2:
            st.sidebar.error("Please select source columns for both files")
            return None
            
        try:
            # Convert embeddings to numpy arrays
            embeddings_1 = np.array([
                eval(emb) if isinstance(emb, str) else emb 
                for emb in files_data["main_df"][embedding_column_1]
            ])
            
            embeddings_2 = np.array([
                eval(emb) if isinstance(emb, str) else emb 
                for emb in files_data["second_df"][embedding_column_2]
            ])
            
            return {
                "mode": "dual",
                "type": "existing",
                "embedding_column_1": embedding_column_1,
                "embedding_column_2": embedding_column_2,
                "embedding_columns_1": [embedding_column_1],
                "embedding_columns_2": [embedding_column_2],
                "source_columns_1": source_columns_1,
                "source_columns_2": source_columns_2,
                "embeddings_1": embeddings_1,
                "embeddings_2": embeddings_2
            }
        except Exception as e:
            st.sidebar.error(f"Error loading embeddings: {str(e)}")
            return None

def generate_embeddings_for_config(files_data: Dict, config: Dict) -> Dict:
    """Generate embeddings based on configuration"""
    try:
        # Create a container for all Streamlit elements
        with st.container():
            model = EmbeddingModel(config["type"], config["model"], config.get("api_key"))
            batch_size = 32  # Optimal batch size for most models
            
            if files_data["mode"] == "single":
                # Prepare text data
                if len(config["source_columns"]) > 1:
                    text_series = combine_text_columns(files_data["main_df"], config["source_columns"])
                else:
                    text_series = files_data["main_df"][config["source_columns"][0]]
                
                texts = list(text_series)
                total = len(texts)
                
                # Create a status container
                status_container = st.empty()
                with status_container.container():
                    st.write("Generating embeddings...")
                    progress_bar = st.progress(0)
                    embeddings_list = []
                    
                    for i in range(0, total, batch_size):
                        batch = texts[i:min(i + batch_size, total)]
                        st.write(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
                        batch_embeddings = model.generate_embeddings(batch)
                        embeddings_list.extend(batch_embeddings)
                        progress = min((i + batch_size) / total, 1.0)
                        progress_bar.progress(progress)
                
                embeddings = np.array(embeddings_list)
                status_container.empty()
                
                # Store results
                embedding_column = f"embeddings_{config['source_columns'][0]}"
                files_data["main_df"][embedding_column] = embeddings.tolist()
                
                result = {
                    "mode": "single",
                    "embedding_columns": [embedding_column],
                    "embedding_column": embedding_column,
                    "embeddings": embeddings,
                    "source_columns": config["source_columns"]
                }
                
            else:  # dual mode
                # Process first file
                if len(config["source_columns_1"]) > 1:
                    text_series_1 = combine_text_columns(files_data["main_df"], config["source_columns_1"])
                else:
                    text_series_1 = files_data["main_df"][config["source_columns_1"][0]]
                
                texts_1 = list(text_series_1)
                total_1 = len(texts_1)
                
                # Process second file
                if len(config["source_columns_2"]) > 1:
                    text_series_2 = combine_text_columns(files_data["second_df"], config["source_columns_2"])
                else:
                    text_series_2 = files_data["second_df"][config["source_columns_2"][0]]
                
                texts_2 = list(text_series_2)
                total_2 = len(texts_2)
                
                # Create status containers for both files
                status_container_1 = st.empty()
                with status_container_1.container():
                    st.write("Processing first file...")
                    progress_bar = st.progress(0)
                    embeddings_list_1 = []
                    
                    for i in range(0, total_1, batch_size):
                        batch = texts_1[i:min(i + batch_size, total_1)]
                        st.write(f"Processing batch {i//batch_size + 1}/{(total_1-1)//batch_size + 1}")
                        batch_embeddings = model.generate_embeddings(batch)
                        embeddings_list_1.extend(batch_embeddings)
                        progress = min((i + batch_size) / total_1, 1.0)
                        progress_bar.progress(progress)
                
                embeddings_1 = np.array(embeddings_list_1)
                status_container_1.empty()
                
                status_container_2 = st.empty()
                with status_container_2.container():
                    st.write("Processing second file...")
                    progress_bar = st.progress(0)
                    embeddings_list_2 = []
                    
                    for i in range(0, total_2, batch_size):
                        batch = texts_2[i:min(i + batch_size, total_2)]
                        st.write(f"Processing batch {i//batch_size + 1}/{(total_2-1)//batch_size + 1}")
                        batch_embeddings = model.generate_embeddings(batch)
                        embeddings_list_2.extend(batch_embeddings)
                        progress = min((i + batch_size) / total_2, 1.0)
                        progress_bar.progress(progress)
                
                embeddings_2 = np.array(embeddings_list_2)
                status_container_2.empty()
                
                # Store results
                embedding_column_1 = f"embeddings_{config['source_columns_1'][0]}"
                embedding_column_2 = f"embeddings_{config['source_columns_2'][0]}"
                
                files_data["main_df"][embedding_column_1] = embeddings_1.tolist()
                files_data["second_df"][embedding_column_2] = embeddings_2.tolist()
                
                result = {
                    "mode": "dual",
                    "embedding_columns_1": [embedding_column_1],
                    "embedding_columns_2": [embedding_column_2],
                    "embedding_column_1": embedding_column_1,
                    "embedding_column_2": embedding_column_2,
                    "embeddings_1": embeddings_1,
                    "embeddings_2": embeddings_2,
                    "source_columns_1": config["source_columns_1"],
                    "source_columns_2": config["source_columns_2"]
                }
            
            # Update session state
            if "files_data" not in st.session_state:
                st.session_state.files_data = {}
            st.session_state.files_data = files_data
            st.session_state.embedding_results = result
            
            return result
                
    except Exception as e:
        st.error(f"Error during embedding generation: {str(e)}")
        st.error(f"Config: {config}")
        return None

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
    # Get model configuration
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
            
        source_columns = st.sidebar.multiselect(
            "Select text columns for embedding",
            options=text_columns,
            help="Choose the columns to generate embeddings from"
        )
        
        if not source_columns:
            st.sidebar.error("Please select at least one column")
            return None
            
        return {
            "mode": "single",
            "source_columns": source_columns,
            **model_config
        }
    
    else:  # dual mode
        # Get text columns from both files
        main_text_columns = [col for col in files_data["main_df"].columns 
                           if files_data["main_df"][col].dtype == 'object']
        second_text_columns = [col for col in files_data["second_df"].columns 
                             if files_data["second_df"][col].dtype == 'object']
        
        if not main_text_columns or not second_text_columns:
            st.sidebar.error("No text columns found in one or both files")
            return None
            
        source_columns_1 = st.sidebar.multiselect(
            "Select text columns from first file",
            options=main_text_columns,
            help="Choose columns from first file"
        )
        
        source_columns_2 = st.sidebar.multiselect(
            "Select text columns from second file",
            options=second_text_columns,
            help="Choose columns from second file"
        )
        
        if not source_columns_1 or not source_columns_2:
            st.sidebar.error("Please select columns from both files")
            return None
            
        return {
            "mode": "dual",
            "source_columns_1": source_columns_1,
            "source_columns_2": source_columns_2,
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
        
        if embedding_results["mode"] == "single":
            # For existing embeddings
            if isinstance(embedding_results.get("embeddings"), np.ndarray):
                embeddings = embedding_results["embeddings"]
                return embeddings, embeddings
            
            # For newly generated embeddings
            elif "embeddings" in embedding_results:
                embeddings = np.array(embedding_results["embeddings"][config["column"]], dtype=np.float32)
                return embeddings, embeddings
                
        else:  # dual mode
            # For existing embeddings
            if "embeddings_1" in embedding_results and "embeddings_2" in embedding_results:
                embeddings1 = embedding_results["embeddings_1"]
                embeddings2 = embedding_results["embeddings_2"]
                return embeddings1, embeddings2
            
            # For newly generated embeddings
            else:
                embeddings1 = np.array(embedding_results["embeddings_1"][config["column1"]], dtype=np.float32)
                embeddings2 = np.array(embedding_results["embeddings_2"][config["column2"]], dtype=np.float32)
                return embeddings1, embeddings2
            
    except Exception as e:
        st.error(f"Error accessing embeddings: {str(e)}")
        if isinstance(embedding_results, dict):
            available_keys = list(embedding_results.keys())
            st.error(f"Available keys in embedding_results: {available_keys}")
            if "embeddings" in embedding_results:
                if isinstance(embedding_results["embeddings"], dict):
                    st.error(f"Available columns: {list(embedding_results['embeddings'].keys())}")
        else:
            st.error(f"embedding_results type: {type(embedding_results)}")
        raise

def get_embeddings_for_clustering(embedding_results: Dict, config: Dict) -> np.ndarray:
    """Get embeddings for clustering analysis"""
    try:
        # Get the selected embedding column from embedding_results
        column = config["column"]
        
        # Convert string embeddings to numpy array
        embeddings_list = []
        for emb_str in st.session_state.files_data["main_df"][column]:
            # Handle string format with commas
            if isinstance(emb_str, str):
                # Remove any brackets and split by comma
                emb_str = emb_str.strip('[]()').split(',')
                # Convert to float array
                emb_array = np.array([float(x.strip()) for x in emb_str])
                embeddings_list.append(emb_array)
            else:
                embeddings_list.append(emb_str)
        
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings
        
    except Exception as e:
        st.error(f"Error getting embeddings for clustering: {str(e)}")
        st.error(f"Sample value: {st.session_state.files_data['main_df'][column].iloc[0]}")
        return None
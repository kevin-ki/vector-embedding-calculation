import streamlit as st
import pandas as pd

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def setup_file_configuration():
    """Initial file configuration setup"""
    st.sidebar.header("1. File Configuration")
    
    file_mode = st.sidebar.radio(
        "Select File Mode",
        options=["Single File", "Two Files"],
        help="Choose whether to work with one or two files"
    )
    
    files_data = {}
    
    try:
        if file_mode == "Single File":
            main_file = st.sidebar.file_uploader(
                "Upload CSV File",
                type="csv",
                help="Upload your CSV file with or without embeddings"
            )
            if main_file:
                files_data["main_df"] = pd.read_csv(main_file)
                files_data["mode"] = "single"
                st.sidebar.success(f"Loaded file with {len(files_data['main_df'])} rows")
        else:  # Two Files
            file1 = st.sidebar.file_uploader(
                "Upload First CSV File",
                type="csv",
                help="Upload your first CSV file with or without embeddings"
            )
            file2 = st.sidebar.file_uploader(
                "Upload Second CSV File",
                type="csv",
                help="Upload your second CSV file with or without embeddings"
            )
            if file1 and file2:
                files_data["main_df"] = pd.read_csv(file1)
                files_data["second_df"] = pd.read_csv(file2)
                files_data["mode"] = "dual"
                st.sidebar.success(f"Loaded files with {len(files_data['main_df'])} and {len(files_data['second_df'])} rows")
    except Exception as e:
        st.error(f"Error loading file(s): {str(e)}")
        return None
        
    return files_data

import numpy as np

def validate_embeddings(embeddings: np.ndarray) -> bool:
    """Validate embedding array format and dimensions"""
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array")
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values")
    return True


def validate_embedding_config(embedding_config):
    """Validate embedding configuration structure"""
    required_keys = ["source", "mode"]
    if not all(key in embedding_config for key in required_keys):
        return False, "Missing required configuration keys"
        
    if embedding_config["source"] == "Use Existing Embeddings":
        if "existing_embeddings" not in embedding_config:
            return False, "Missing existing embeddings configuration"
            
        existing_config = embedding_config["existing_embeddings"]
        if existing_config["mode"] == "single":
            if not all(key in existing_config for key in ["embedding_columns", "embeddings"]):
                return False, "Invalid single file embedding configuration"
        else:  # dual mode
            if not all(key in existing_config for key in 
                      ["embedding_columns_1", "embedding_columns_2", "embeddings_1", "embeddings_2"]):
                return False, "Invalid dual file embedding configuration"
                
    return True, "Configuration valid"



def validate_embeddings_format(embeddings):
    """Validate embedding array format"""
    try:
        if not isinstance(embeddings, dict):
            return False, "Embeddings must be a dictionary"
            
        for col, emb in embeddings.items():
            if not isinstance(emb, np.ndarray):
                return False, f"Embeddings for column {col} must be numpy array"
            if len(emb.shape) != 2:
                return False, f"Embeddings for column {col} must be 2D array"
            if np.isnan(emb).any():
                return False, f"Embeddings for column {col} contain NaN values"
                
        return True, "Embeddings valid"
    except Exception as e:
        return False, f"Error validating embeddings: {str(e)}" 
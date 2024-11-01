
# Available embedding models for each provider
OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small": "Latest small model, good balance of performance and cost",
    "text-embedding-3-large": "Latest large model, highest performance"
}

VOYAGE_EMBEDDING_MODELS = {
    "voyage-2": "Base model",
    "voyage-3": "Latest model with improved performance"
}

SENTENCE_TRANSFORMER_MODELS = {
    "all-MiniLM-L6-v2": "Fast and efficient general-purpose model",
    "all-mpnet-base-v2": "Higher quality, slower model"
}

GOOGLE_EMBEDDING_MODELS = {
    "models/text-embedding-004": "Latest model, good balance of performance and cost",
    "models/embedding-001": "General purpose embedding model"
}

# Default model selections
DEFAULT_OPENAI_MODEL = list(OPENAI_EMBEDDING_MODELS.keys())[0]
DEFAULT_VOYAGE_MODEL = list(VOYAGE_EMBEDDING_MODELS.keys())[0]
DEFAULT_SENTENCE_TRANSFORMER_MODEL = list(SENTENCE_TRANSFORMER_MODELS.keys())[0]
DEFAULT_GOOGLE_MODEL = list(GOOGLE_EMBEDDING_MODELS.keys())[0]
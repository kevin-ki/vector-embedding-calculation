# Vector Embedding Analysis Tool

A powerful tool for analyzing vector embeddings through similarity calculations and clustering algorithms.

## Comparison of Similarity and Neighbor Algorithms

| Method | Description | Advantages | Disadvantages | Necessary Parameters |
|--------|-------------|------------|---------------|---------------------|
| Cosine Similarity | Calculates the angle between vectors | - Easy to calculate and understand<br>- Works well with high-dimensional data<br>- Scale-invariant | - Does not consider magnitude<br>- May miss important magnitude differences | None |
| FAISS Search | Facebook AI's similarity search | - Extremely fast on large datasets<br>- Memory efficient<br>- Supports GPU acceleration | - Approximate results<br>- Requires careful index configuration | - Number of neighbors (k)<br>- Index type |

## Application of k-Values

| Use Case | k-Value | Explanation |
|----------|---------|-------------|
| Direct Mapping | 1 | Find single best match for each item |
| Alternative Suggestions | 3-5 | Identify multiple similar items |
| Content Clustering | 5-10 | Group similar content together |
| Duplicate Detection | 1-3 | Identify potential duplicates |

## Comparison of Clustering Algorithms

| Method | Description | Advantages | Disadvantages | Necessary Parameters |
|--------|-------------|------------|---------------|---------------------|
| K-Means | Partitions data into k clusters | - Fast and simple<br>- Works well with spherical clusters | - Requires predefined k<br>- Sensitive to outliers | - Number of clusters (k) |
| DBSCAN | Density-based spatial clustering | - Finds arbitrary shapes<br>- Handles outliers well | - Struggles with varying densities<br>- Sensitive to parameters | - Epsilon (ε)<br>- Min samples |
| HDBSCAN | Hierarchical density clustering | - Adaptive to different densities<br>- No epsilon parameter needed | - Slower than DBSCAN<br>- More complex implementation | - Min cluster size<br>- Min samples |
| OPTICS | Ordering points to identify clustering structure | - Handles varying densities<br>- Creates reachability plot | - Slower than other methods<br>- Complex parameter tuning | - Min samples<br>- Max epsilon |

## Features

1. **File Handling**
   - Single or dual file mode
   - CSV file support
   - Column selection and preview

2. **Embedding Generation**
   - Multiple embedding models
   - Column combination options
   - Batch processing

3. **Analysis Options**
   - Similarity Calculation
     - Cosine Similarity
     - FAISS Search
   - Clustering
     - K-Means
     - DBSCAN
     - HDBSCAN
     - OPTICS

4. **Visualization**
   - Interactive heatmaps
   - Cluster visualizations
   - Dimensionality reduction plots
   - Export options

5. **Export Capabilities**
   - Raw results
   - Enhanced DataFrames
   - Visualization exports

## Before Using This App

### Prerequisites

1. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. **File Format**
   - CSV files with UTF-8 encoding
   - Clean, preprocessed text data
   - Consistent column names

2. **Memory Considerations**
   - Large datasets may require batch processing
   - Monitor memory usage with large embeddings

### Getting Started

1. Clone the repository
2. Install dependencies
3. Set up environment variables
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Best Practices

1. Start with small datasets for testing
2. Monitor memory usage
3. Use appropriate k-values for your use case
4. Save results frequently
5. Validate clustering results

## Model Providers & API Keys

| Provider | API Key Required | Registration/Documentation |
|----------|-----------------|---------------------------|
| Sentence Transformers | No | [Documentation](https://www.sbert.net/) |
| OpenAI | Yes | [Get API Key](https://platform.openai.com/api-keys) |
| Voyage AI | Yes | [Get API Key](https://dash.voyageai.com/api-keys) |
| Google | Yes | [Get API Key](https://aistudio.google.com/app/apikey) |

### API Key Setup

1. **Sentence Transformers**
   - No API key required
   - Open-source and locally run models
   - [Browse available models](https://huggingface.co/sentence-transformers)

2. **OpenAI**
   - Create account at [OpenAI Platform](https://platform.openai.com)
   - Navigate to API Keys section
   - Create new secret key
   - Store securely and never share publicly

3. **Voyage AI**
   - Sign up at [Voyage AI](https://www.voyageai.com)
   - Request API access
   - Generate API key from dashboard
   - Store securely

4. **Google AI**
   - Create Google Cloud account
   - Enable Generative AI API
   - Create credentials in [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Store API key securely

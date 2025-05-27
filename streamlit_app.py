# streamlit_app.py
import streamlit as st
import os
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.data_connectors import (
    get_documents_from_s3, 
    get_documents_from_confluence,
    get_documents_from_sharepoint
)

# --- Page Configuration ---
st.set_page_config(page_title="HippoRAG Interactive Demo", layout="wide")

# --- Helper Functions ---
@st.cache_resource # Cache the HippoRAG instance for performance
def get_hipporag_instance(config_dict):
    config = BaseConfig(**config_dict)
    return HippoRAG(global_config=config)

def display_results(query_solutions):
    if not query_solutions:
        st.warning("No results to display.")
        return

    for i, solution in enumerate(query_solutions):
        st.subheader(f"Result for Query: \"{solution.question}\"")
        st.markdown(f"**Answer:** {solution.answer}")
        with st.expander("Show Retrieved Documents"):
            if solution.docs:
                for j, doc_content in enumerate(solution.docs):
                    st.text_area(f"Document {j+1}", doc_content, height=150, key=f"doc_{i}_{j}")
            else:
                st.write("No documents were retrieved for this query.")
        st.divider()

# --- Application State ---
if 'hipporag_instance' not in st.session_state:
    st.session_state.hipporag_instance = None
if 'indexed_docs_count' not in st.session_state:
    st.session_state.indexed_docs_count = 0
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'current_docs' not in st.session_state:
    st.session_state.current_docs = []


# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")

# LLM Provider and API Keys
st.sidebar.subheader("LLM Configuration")
llm_provider = st.sidebar.selectbox("Select LLM Provider", ["OpenAI", "Azure OpenAI", "OpenRouter"], key="llm_provider")

api_key_placeholder = {
    "OpenAI": "sk-...",
    "Azure OpenAI": "Azure API Key", # Azure also needs endpoint
    "OpenRouter": "ork-..." 
}

api_key = st.sidebar.text_input(f"{llm_provider} API Key", type="password", 
                                placeholder=api_key_placeholder.get(llm_provider, ""), 
                                key="api_key",
                                help="Leave blank to use environment variables (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY).")

llm_model_name = st.sidebar.text_input("LLM Model Name", value="gpt-4o-mini", key="llm_model_name",
                                       help="e.g., gpt-3.5-turbo, gpt-4o-mini, openai/gpt-4o (for OpenRouter)")
azure_endpoint_val = None
if llm_provider == "Azure OpenAI":
    azure_endpoint_val = st.sidebar.text_input("Azure OpenAI Endpoint", key="azure_endpoint", 
                                               placeholder="https://your-resource.openai.azure.com/")

# Embedding Model
st.sidebar.subheader("Embedding Model")
embedding_model_name = st.sidebar.text_input("Embedding Model Name", value="GritLM/GritLM-7B", key="embedding_model",
                                             help="e.g., GritLM/GritLM-7B, NV-Embed-QA, baai/bge-large-en-v1.5")
embedding_base_url = st.sidebar.text_input("Embedding Model Base URL (Optional)", key="embedding_base_url",
                                           help="For self-hosted embedding models.")

# Data Source Selection
st.sidebar.subheader("📚 Data Source")
data_source_type = st.sidebar.selectbox(
    "Choose Data Source",
    ["Text Input", "S3 Bucket", "Confluence", "SharePoint"],
    key="data_source_type"
)

# --- Data Source Configuration (Dynamic in Sidebar) ---
s3_bucket, s3_prefix, s3_aws_access_key, s3_aws_secret_key, s3_region = "", "", "", "", ""
confluence_url, confluence_space, confluence_user, confluence_token, confluence_cql = "", "", "", "", ""
sharepoint_site_url, sharepoint_library, sharepoint_folder, sharepoint_user, sharepoint_pass = "", "", "", "", ""

if data_source_type == "S3 Bucket":
    st.sidebar.markdown("--- \n**S3 Configuration**")
    s3_bucket = st.sidebar.text_input("S3 Bucket Name", key="s3_bucket")
    s3_prefix = st.sidebar.text_input("S3 Prefix (Optional)", key="s3_prefix")
    s3_aws_access_key = st.sidebar.text_input("AWS Access Key ID (Optional)", key="s3_key", type="password", help="Uses env vars if blank")
    s3_aws_secret_key = st.sidebar.text_input("AWS Secret Access Key (Optional)", key="s3_secret", type="password", help="Uses env vars if blank")
    s3_region = st.sidebar.text_input("AWS Region (Optional)", key="s3_region", help="e.g., us-east-1. Uses env vars/default if blank")
elif data_source_type == "Confluence":
    st.sidebar.markdown("--- \n**Confluence Configuration**")
    confluence_url = st.sidebar.text_input("Confluence URL", key="confluence_url", placeholder="https://yourcompany.atlassian.net/wiki")
    confluence_space = st.sidebar.text_input("Space Key (Optional, if not using CQL)", key="confluence_space")
    confluence_cql = st.sidebar.text_input("CQL Query (Optional, if not using Space)", key="confluence_cql", placeholder='type=page and label="yourlabel"')
    confluence_user = st.sidebar.text_input("Confluence Username/Email (Optional)", key="confluence_user", type="password", help="Uses env vars if blank")
    confluence_token = st.sidebar.text_input("Confluence API Token (Optional)", key="confluence_token", type="password", help="Uses env vars if blank")
elif data_source_type == "SharePoint":
    st.sidebar.markdown("--- \n**SharePoint Configuration**")
    sharepoint_site_url = st.sidebar.text_input("SharePoint Site URL", key="sharepoint_site", placeholder="https://yourtenant.sharepoint.com/sites/YourSite")
    sharepoint_library = st.sidebar.text_input("Document Library Name", value="Shared Documents", key="sharepoint_library")
    sharepoint_folder = st.sidebar.text_input("Folder Path (Optional)", key="sharepoint_folder", placeholder="MyFolder/SubFolder")
    sharepoint_user = st.sidebar.text_input("SharePoint Username (Optional)", key="sharepoint_user", type="password", help="Uses env vars if blank")
    sharepoint_pass = st.sidebar.text_input("SharePoint Password (Optional)", key="sharepoint_pass", type="password", help="Uses env vars if blank")


# --- Main Application Area ---
st.title("🦛 HippoRAG Interactive Demo")
st.markdown("Configure your RAG pipeline using the sidebar, input your documents or connect to a data source, then ask questions!")

# --- Document Input Area ---
st.header("1. Ingest Documents")

if data_source_type == "Text Input":
    with st.form("doc_input_form"):
        raw_docs_input = st.text_area("Enter documents (one per line or separated by double newlines):", height=200, key="raw_docs")
        submitted_docs = st.form_submit_button("Load Documents from Text")
        if submitted_docs and raw_docs_input:
            docs_list = [doc.strip() for doc in raw_docs_input.split('\n\n') if doc.strip()]
            if not docs_list: # Try splitting by single newline if double fails
                 docs_list = [doc.strip() for doc in raw_docs_input.split('\n') if doc.strip()]
            st.session_state.current_docs = docs_list
            st.success(f"Loaded {len(st.session_state.current_docs)} documents from text input.")
            st.session_state.indexed_docs_count = 0 # Reset index count
            st.session_state.query_results = None
elif st.button(f"Load Documents from {data_source_type}"):
    st.session_state.current_docs = []
    with st.spinner(f"Fetching from {data_source_type}..."):
        if data_source_type == "S3 Bucket":
            if not s3_bucket:
                st.error("S3 Bucket Name is required.")
            else:
                st.session_state.current_docs = get_documents_from_s3(
                    aws_access_key_id=s3_aws_access_key or None, # Pass None if empty to use env vars
                    aws_secret_access_key=s3_aws_secret_key or None,
                    region_name=s3_region or None,
                    bucket_name=s3_bucket,
                    prefix=s3_prefix
                )
        elif data_source_type == "Confluence":
            if not confluence_url or (not confluence_space and not confluence_cql):
                st.error("Confluence URL and either Space Key or CQL query are required.")
            else:
                st.session_state.current_docs = get_documents_from_confluence(
                    confluence_url=confluence_url,
                    username=confluence_user or None,
                    api_token=confluence_token or None,
                    space_key=confluence_space or None,
                    cql=confluence_cql or None
                )
        elif data_source_type == "SharePoint":
            if not sharepoint_site_url:
                st.error("SharePoint Site URL is required.")
            else:
                st.session_state.current_docs = get_documents_from_sharepoint(
                    sharepoint_url=sharepoint_site_url,
                    username=sharepoint_user or None,
                    password=sharepoint_pass or None,
                    document_library_name=sharepoint_library,
                    folder_path=sharepoint_folder or None
                )
    if st.session_state.current_docs:
        st.success(f"Successfully loaded {len(st.session_state.current_docs)} documents from {data_source_type}.")
        st.session_state.indexed_docs_count = 0 # Reset index count
        st.session_state.query_results = None
    else:
        st.error(f"Failed to load documents from {data_source_type} or no documents found.")

if st.session_state.current_docs:
    st.info(f"{len(st.session_state.current_docs)} documents loaded. Ready for indexing.")
    with st.expander("View Loaded Documents (First 500 chars of first 5 docs)"):
        for i, doc in enumerate(st.session_state.current_docs[:5]):
            st.text_area(f"Document {i+1}", doc[:500] + "..." if len(doc) > 500 else doc, height=100, key=f"loaded_doc_preview_{i}", disabled=True)


# --- Indexing Area ---
if st.session_state.current_docs:
    if st.button("Index Loaded Documents", key="index_button"):
        if not st.session_state.current_docs:
            st.warning("No documents loaded to index.")
        else:
            # Prepare BaseConfig dict
            config_updates = {
                "llm_name": llm_model_name,
                "embedding_model_name": embedding_model_name,
                "save_dir": os.path.join("outputs", "streamlit_demo") # Specific save dir for demo
            }
            if api_key: # Only override if user provided one
                if llm_provider == "OpenAI":
                    config_updates["openai_api_key"] = api_key
                elif llm_provider == "OpenRouter":
                    config_updates["openrouter_api_key"] = api_key
                    config_updates["llm_base_url"] = "https://openrouter.ai/api/v1"
                elif llm_provider == "Azure OpenAI":
                    config_updates["azure_api_key"] = api_key # Assuming BaseConfig uses this name
                    config_updates["azure_endpoint"] = azure_endpoint_val
            
            if llm_provider == "OpenRouter" and "llm_base_url" not in config_updates : # Set if not set by API key logic
                 config_updates["llm_base_url"] = "https://openrouter.ai/api/v1"
            
            if embedding_base_url:
                config_updates["embedding_base_url"] = embedding_base_url
            
            try:
                with st.spinner("Initializing HippoRAG and indexing documents... This may take a while."):
                    # Create a BaseConfig object and update it
                    base_cfg = BaseConfig()
                    for key_conf, value_conf in config_updates.items():
                        if hasattr(base_cfg, key_conf):
                            setattr(base_cfg, key_conf, value_conf)
                        else:
                            st.warning(f"Config key {key_conf} not found in BaseConfig. Skipping.")
                    
                    # Ensure API keys from env are loaded if not provided by user
                    # BaseConfig __post_init__ should handle this if api_key fields are None
                    if not api_key: # If user didn't provide, ensure BaseConfig loads from env
                        if llm_provider == "OpenAI" and not base_cfg.openai_api_key:
                             base_cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
                        elif llm_provider == "OpenRouter" and not base_cfg.openrouter_api_key:
                             base_cfg.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                        elif llm_provider == "Azure OpenAI" and not base_cfg.azure_api_key:
                             base_cfg.azure_api_key = os.getenv("AZURE_API_KEY") # Or specific Azure var
                    
                    # Log effective config for debugging
                    st.write("Effective HippoRAG Config (excluding API keys for security):")
                    config_display = {k:v for k,v in base_cfg.__dict__.items() if 'api_key' not in k and 'token' not in k}
                    st.json(config_display)

                    st.session_state.hipporag_instance = HippoRAG(global_config=base_cfg)
                    st.session_state.hipporag_instance.index(docs=st.session_state.current_docs)
                    st.session_state.indexed_docs_count = len(st.session_state.current_docs)
                st.success(f"Successfully indexed {st.session_state.indexed_docs_count} documents!")
                st.session_state.query_results = None # Clear previous results
            except Exception as e:
                st.error(f"Error during indexing: {e}")
                # import traceback
                # st.text(traceback.format_exc())


# --- Query Area ---
st.header("2. Ask a Question")
if st.session_state.indexed_docs_count > 0:
    st.success(f"{st.session_state.indexed_docs_count} documents are indexed and ready for querying.")
    query_text = st.text_input("Enter your query:", key="query_text")
    if st.button("Get Answer", key="query_button"):
        if not query_text:
            st.warning("Please enter a query.")
        elif not st.session_state.hipporag_instance:
            st.warning("HippoRAG instance not initialized. Please index documents first.")
        else:
            with st.spinner("Retrieving answer..."):
                try:
                    # The rag_qa method expects a list of queries
                    results, _, _ = st.session_state.hipporag_instance.rag_qa(queries=[query_text])
                    st.session_state.query_results = results
                except Exception as e:
                    st.error(f"Error during querying: {e}")
                    # import traceback
                    # st.text(traceback.format_exc())
else:
    st.info("Please load and index some documents to enable querying.")


# --- Results Display Area ---
if st.session_state.query_results:
    st.header("3. Results")
    display_results(st.session_state.query_results)

st.sidebar.markdown("---")
st.sidebar.info("HippoRAG Streamlit Demo")

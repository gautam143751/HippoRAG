import os
import logging
from typing import List, Dict, Any, Optional # Removed unused types like Union, Set, Tuple, Literal

# Removed numpy, tqdm, pandas, copy.deepcopy as they are likely handled by specific stores or not needed here.
# Removed compute_mdhash_id, NerRawOutput, TripleRawOutput as hashing is delegated.

from .vector_stores.base import VectorStore
from .vector_stores.parquet import ParquetVectorStore
from .vector_stores.supabase import SupabaseVectorStore
from .vector_stores.pinecone_db import PineconeVectorStore
# Assuming BaseEmbeddingModel is needed for type hinting embedding_model if it's accessed directly.
# from .embedding_model.base import BaseEmbeddingModel 

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, 
                 embedding_model: Any, # Or BaseEmbeddingModel if type hinting
                 namespace: str,
                 vector_store_type: str, 
                 vector_store_config: Dict[str, Any],
                 batch_size: int = 32, # Kept as it's passed to stores
                 ):
        """
        Initializes the EmbeddingStore which acts as a factory and delegate for various vector stores.

        Parameters:
        embedding_model: The model used for embeddings.
        namespace: A unique identifier for data segregation, passed to the underlying store.
        vector_store_type: Type of vector store to use (e.g., 'parquet', 'supabase', 'pinecone').
        vector_store_config: Configuration dictionary for the chosen vector store.
        batch_size: The batch size used for processing, passed to the underlying store.
        """
        self.embedding_model = embedding_model # Stored if needed directly, else just passed
        self.namespace = namespace # Stored if needed directly, else just passed
        self.batch_size = batch_size # Stored as it's passed to stores

        logger.info(f"Initializing EmbeddingStore with vector_store_type: {vector_store_type}")

        if vector_store_type == 'parquet':
            db_directory = vector_store_config.get("db_directory", "./default_parquet_db/")
            # ParquetVectorStore expects 'db_directory', not 'db_filename'
            self.store: VectorStore = ParquetVectorStore(
                embedding_model=self.embedding_model, 
                db_directory=db_directory, 
                batch_size=self.batch_size, 
                namespace=self.namespace
            )
        elif vector_store_type == 'supabase':
            supabase_url = vector_store_config.get("supabase_url")
            supabase_key = vector_store_config.get("supabase_key")
            table_name = vector_store_config.get("table_name", "embeddings") # Default table name
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be provided for Supabase vector store.")
            self.store: VectorStore = SupabaseVectorStore(
                embedding_model=self.embedding_model,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                table_name=table_name,
                batch_size=self.batch_size,
                namespace=self.namespace
            )
        elif vector_store_type == 'pinecone':
            pinecone_api_key = vector_store_config.get("pinecone_api_key")
            pinecone_environment = vector_store_config.get("pinecone_environment")
            index_name = vector_store_config.get("index_name", "hipporag-index") # Default index name
            if not pinecone_api_key or not pinecone_environment:
                raise ValueError("Pinecone API key and environment must be provided for Pinecone vector store.")
            # PineconeVectorStore expects the actual embedding_model object
            self.store: VectorStore = PineconeVectorStore(
                embedding_model=self.embedding_model, 
                pinecone_api_key=pinecone_api_key,
                pinecone_environment=pinecone_environment,
                index_name=index_name,
                batch_size=self.batch_size,
                namespace=self.namespace
            )
        else:
            raise ValueError(f"Unknown vector_store_type: {vector_store_type}")

        logger.info(f"Successfully initialized backend vector store: {vector_store_type}")

    # Methods to be delegated to self.store
    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        return self.store.get_missing_string_hash_ids(texts)

    def insert_strings(self, texts: List[str]):
        # Return type is None as per VectorStore.insert_strings
        self.store.insert_strings(texts)
        return # Explicitly return None

    def delete(self, hash_ids: List[str]):
        # Return type is None
        self.store.delete(hash_ids)
        return # Explicitly return None

    def get_row(self, hash_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_row(hash_id)

    # get_hash_id was specific to the old implementation's text_to_hash_id map
    # This functionality is now internal to each VectorStore or not directly exposed
    # def get_hash_id(self, text: str) -> Optional[str]:
    #     # This would require the specific store to implement such a method if needed.
    #     # For now, removing as it's not part of the VectorStore ABC.
    #     # if hasattr(self.store, 'get_hash_id_for_text'): # Example of conditional delegation
    #     #     return self.store.get_hash_id_for_text(text)
    #     logger.warning("'get_hash_id' is not a standard VectorStore method and has been removed.")
    #     return None

    def get_rows(self, hash_ids: List[str], dtype: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        # The 'dtype' parameter was specific to the old Parquet implementation's get_embeddings.
        # For get_rows, it's not standard. VectorStore.get_rows doesn't specify dtype.
        # If concrete stores need dtype for their get_rows, they should handle it or it should be part of their config.
        if dtype is not None:
            logger.warning("The 'dtype' parameter for get_rows is deprecated in EmbeddingStore and may not be used by the underlying vector store.")
        return self.store.get_rows(hash_ids)

    def get_all_ids(self) -> List[str]:
        return self.store.get_all_ids()

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        return self.store.get_all_id_to_rows()

    def get_all_texts(self) -> List[str]: # VectorStore ABC defines this as returning List[str]
        return self.store.get_all_texts()

    def get_embedding(self, hash_id: str, dtype: Optional[Any] = None) -> Optional[np.ndarray]:
        # The 'dtype' parameter for get_embedding might be specific to some stores.
        # The VectorStore ABC for get_embedding is (self, hash_id: str) -> Any.
        # ParquetVectorStore implemented it with dtype. Supabase/Pinecone return np.ndarray already.
        # For consistency, if a store supports dtype, it should be passed.
        # We can pass it as a kwarg if the underlying store supports it.
        if hasattr(self.store, 'get_embedding') and 'dtype' in self.store.get_embedding.__code__.co_varnames:
             return self.store.get_embedding(hash_id, dtype=dtype) # type: ignore
        else:
            if dtype is not None:
                 logger.warning(f"Store {type(self.store)} does not support 'dtype' for get_embedding. Ignoring.")
            return self.store.get_embedding(hash_id)


    def get_embeddings(self, hash_ids: List[str], dtype: Optional[Any] = None) -> List[Optional[Any]]: # VectorStore.get_embeddings returns List[Any]
        # Similar to get_embedding, handle dtype if the store supports it.
        if hasattr(self.store, 'get_embeddings') and 'dtype' in self.store.get_embeddings.__code__.co_varnames:
            return self.store.get_embeddings(hash_ids, dtype=dtype) # type: ignore
        else:
            if dtype is not None:
                logger.warning(f"Store {type(self.store)} does not support 'dtype' for get_embeddings. Ignoring.")
            return self.store.get_embeddings(hash_ids)
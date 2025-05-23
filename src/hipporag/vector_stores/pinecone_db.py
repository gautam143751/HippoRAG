import os
import logging
from typing import List, Any, Dict, Optional

import numpy as np
import pinecone

from ..base import VectorStore
from ..utils.misc_utils import compute_mdhash_id
from ..embedding_model.base import BaseEmbeddingModel # To type hint embedding_model

logger = logging.getLogger(__name__)

class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_environment: str,
        embedding_model: Optional[BaseEmbeddingModel], # Allow None
        index_name: str,
        namespace: str, # Pinecone namespace
        vector_store_config: Optional[Dict[str, Any]] = None, # Added for dimension
        batch_size: int = 100, # Pinecone recommends batch sizes up to 100 for upsert
        # metric: str = "cosine", # Pinecone default is cosine, can be configured at index creation
    ):
        if not pinecone_api_key or not pinecone_environment:
            raise ValueError("Pinecone API key and environment must be provided.")
        if not index_name:
            raise ValueError("Pinecone index name must be provided.")
        # Embedding model can be None, dimension will be sought from config

        self.embedding_model = embedding_model
        self.namespace = namespace
        self.batch_size = batch_size
        self.index_name = index_name

        # Initialize Pinecone connection
        # pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        # Using the new Pinecone client v3.x.x syntax
        self.pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)


        # Get or create Pinecone index
        self.dimension = None
        if self.embedding_model:
            self.dimension = self.embedding_model.get_dimension()
        elif vector_store_config and 'dimension' in vector_store_config:
            self.dimension = vector_store_config['dimension']
            logger.info(f"Using dimension {self.dimension} from vector_store_config for Pinecone index '{index_name}'.")
        else:
            logger.warning(
                f"PineconeVectorStore: embedding_model is None and 'dimension' not found in vector_store_config. "
                f"Index '{index_name}' might not be created or might be unusable if it needs creation."
            )
            # Index creation might fail if dimension is None and index doesn't exist.
            # If index exists, operations might still work if dimension matches.

        active_indexes_response = self.pinecone_client.list_indexes()
        active_indexes = [idx_spec.name for idx_spec in active_indexes_response] # Updated to access name attribute

        if index_name not in active_indexes:
            if self.dimension:
                logger.info(f"Pinecone index '{index_name}' not found. Creating a new one with dimension {self.dimension}.")
                try:
                    self.pinecone_client.create_index(
                        name=index_name, 
                        dimension=self.dimension, 
                        metric='cosine', 
                    )
                    logger.info(f"Pinecone index '{index_name}' created successfully.")
                except pinecone.core.client.exceptions.ApiException as e:
                    if e.status == 409: 
                        logger.info(f"Pinecone index '{index_name}' already exists (possibly created by another process).")
                    else:
                        logger.error(f"Error creating Pinecone index '{index_name}': {e}")
                        raise
            else:
                logger.error(f"Cannot create Pinecone index '{index_name}' because dimension is unknown.")
                # Depending on strictness, could raise error here. For now, it will fail if it tries to use self.index.
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'.")

        self.index = self.pinecone_client.Index(index_name)
        logger.info(f"PineconeVectorStore initialized for index '{index_name}' and namespace '{self.namespace}'.")
        # Describe index stats to confirm connection and get initial details
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get Pinecone index stats for '{index_name}': {e}")


    def _load_data(self):
        logger.info(f"[_load_data] Pinecone handles data persistence. Index '{self.index_name}' is assumed to be managed by Pinecone.")
        pass

    def _save_data(self):
        logger.info(f"[_save_data] Pinecone handles data persistence directly. No file saving operation performed.")
        pass

    def _upsert(self, hash_ids: List[str], texts: List[str], embeddings: List[List[float]]):
        if not hash_ids:
            return

        vectors_to_upsert = []
        for h_id, text, emb in zip(hash_ids, texts, embeddings):
            # Ensure embedding is a list of floats, not np.ndarray
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb) 

            vectors_to_upsert.append({
                "id": h_id,
                "values": emb_list,
                "metadata": {"content": text, "hash_id": h_id} # Storing text in metadata
            })

        if not vectors_to_upsert:
            return

        try:
            for i in range(0, len(vectors_to_upsert), self.batch_size):
                batch_vectors = vectors_to_upsert[i:i + self.batch_size]
                self.index.upsert(vectors=batch_vectors, namespace=self.namespace)
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors into Pinecone index '{self.index_name}' namespace '{self.namespace}'.")
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            # Potentially re-raise or handle more gracefully
            raise

    def insert_strings(self, texts: List[str]):
        if not self.embedding_model:
            logger.error("Cannot insert strings: embedding_model is None.")
            raise ValueError("Embedding model is not available for PineconeVectorStore to insert strings.")
        if not texts:
            return

        # 1. Generate hash_id for each text
        # Hash ID includes namespace prefix, consistent with other stores
        # prospective_hash_ids = [compute_mdhash_id(text, prefix=f"{self.namespace}-") for text in texts]
        # However, Pinecone's namespace is a separate parameter, so hash_id should be unique without internal namespacing.
        # The self.namespace field of the class is used in Pinecone API calls.
        # Let's use compute_mdhash_id without a prefix here, assuming hash_id itself is the unique key.
        # If global uniqueness across namespaces in the same index is needed, prefixing might be reconsidered.
        # For now, relying on Pinecone's namespace for separation.

        # 2. Check which texts/hash_ids are missing
        # The base class definition for get_missing_string_hash_ids is `(self, texts: List[str]) -> Dict[str, Dict[str, Any]]`
        # This method will internally generate hash_ids from texts and check their existence in Pinecone.
        missing_info: Dict[str, Dict[str, Any]] = self.get_missing_string_hash_ids(texts)

        if not missing_info:
            logger.info(f"No new strings to insert into Pinecone index '{self.index_name}' namespace '{self.namespace}'. All provided texts already exist.")
            return

        new_hash_ids_to_insert = list(missing_info.keys())
        new_texts_to_insert = [missing_info[h_id]["content"] for h_id in new_hash_ids_to_insert]

        if not new_texts_to_insert: # Should not happen if missing_info is not empty
            return

        logger.info(f"Found {len(new_texts_to_insert)} new strings to insert into Pinecone.")

        # 3. Generate embeddings for the new texts
        # The embedding_model.batch_encode should return List[np.ndarray] or similar
        new_embeddings_np = self.embedding_model.batch_encode(new_texts_to_insert, batch_size=self.batch_size)
        
        # Convert embeddings to List[List[float]] for _upsert
        new_embeddings_list = [emb.tolist() for emb in new_embeddings_np]

        # 4. Call _upsert to store the new hash_ids, texts, and embeddings
        self._upsert(new_hash_ids_to_insert, new_texts_to_insert, new_embeddings_list)
        
        # 5. Return None
        return

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        if not texts:
            return {}

        # 1. Generate hash_id for each text
        # Using the raw text for hash generation, namespace is handled by Pinecone client
        prospective_hash_id_to_text_map: Dict[str, str] = {
            compute_mdhash_id(text): text for text in texts # No prefix for Pinecone hash_id
        }
        
        if not prospective_hash_id_to_text_map:
            return {}

        prospective_hash_ids = list(prospective_hash_id_to_text_map.keys())

        # 2. Query Pinecone to find which hash_ids already exist
        missing_texts_info: Dict[str, Dict[str, Any]] = {}
        try:
            # Pinecone's fetch returns only existing vectors.
            # We need to fetch in batches if prospective_hash_ids is very large,
            # as fetch has limits (e.g., 1000 IDs per call).
            # For simplicity here, assuming the list is not excessively long.
            # A more robust implementation would batch these calls.
            
            # Initialize all as missing
            for h_id, text_content in prospective_hash_id_to_text_map.items():
                missing_texts_info[h_id] = {"hash_id": h_id, "content": text_content}

            # Fetch existing IDs and remove them from missing_texts_info
            # Batching fetch requests for robustness if many IDs
            for i in range(0, len(prospective_hash_ids), self.batch_size): # Using self.batch_size for fetch too
                batch_ids_to_fetch = prospective_hash_ids[i:i + self.batch_size]
                if not batch_ids_to_fetch: continue

                fetch_response = self.index.fetch(ids=batch_ids_to_fetch, namespace=self.namespace)
                if fetch_response and fetch_response.vectors:
                    for h_id_fetched in fetch_response.vectors.keys():
                        if h_id_fetched in missing_texts_info:
                            del missing_texts_info[h_id_fetched] # Remove if found

        except Exception as e:
            logger.error(f"Exception fetching existing hash_ids from Pinecone: {e}")
            # If fetch fails, we might incorrectly mark all as missing.
            # Depending on desired behavior, might re-raise or return empty.
            # For now, it will return all as missing if an error occurs during fetch,
            # which could lead to re-embedding. A more sophisticated retry or error handling
            # for partial success might be needed in a production system.
            # Or, to be safe, return empty to prevent re-embedding on transient errors:
            # return {} 
            # Let's stick to returning current missing_texts_info which would be all if fetch fails.

        return missing_texts_info


    def get_row(self, hash_id: str) -> Optional[Dict[str, Any]]:
        if not hash_id:
            return None
        try:
            fetch_response = self.index.fetch(ids=[hash_id], namespace=self.namespace)
            if fetch_response and fetch_response.vectors and hash_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[hash_id]
                if vector_data.metadata and "content" in vector_data.metadata:
                    return {"hash_id": hash_id, "content": vector_data.metadata["content"]}
                else:
                    logger.warning(f"Content not found in metadata for hash_id '{hash_id}' in Pinecone.")
                    # If content is essential, return None or a row with None content
                    return {"hash_id": hash_id, "content": None} 
            else:
                logger.info(f"Vector for hash_id '{hash_id}' not found in Pinecone namespace '{self.namespace}'.")
                return None
        except Exception as e:
            logger.error(f"Error fetching row for hash_id '{hash_id}' from Pinecone: {e}")
            return None

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not hash_ids:
            return {}
        
        rows_dict: Dict[str, Dict[str, Any]] = {}
        try:
            # Batching fetch requests if hash_ids list is very long
            for i in range(0, len(hash_ids), self.batch_size): # Using self.batch_size for fetch batching
                batch_ids_to_fetch = hash_ids[i:i + self.batch_size]
                if not batch_ids_to_fetch: continue

                fetch_response = self.index.fetch(ids=batch_ids_to_fetch, namespace=self.namespace)
                if fetch_response and fetch_response.vectors:
                    for h_id, vector_data in fetch_response.vectors.items():
                        if vector_data.metadata and "content" in vector_data.metadata:
                            rows_dict[h_id] = {"hash_id": h_id, "content": vector_data.metadata["content"]}
                        else:
                            logger.warning(f"Content not found in metadata for hash_id '{h_id}' in Pinecone during get_rows.")
                            rows_dict[h_id] = {"hash_id": h_id, "content": None}
        except Exception as e:
            logger.error(f"Error fetching rows for hash_ids from Pinecone: {e}")
            # Depending on requirements, might return partially fetched data or an empty dict on error.
            # For now, returns what was fetched before an error.
        return rows_dict

    def get_all_ids(self) -> List[str]:
        # Implementation will follow
        logger.warning("Fetching all IDs from Pinecone can be inefficient and is not directly supported. This method may be removed or changed.")
        raise NotImplementedError("Fetching all IDs from Pinecone is not efficiently supported.")

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        # Implementation will follow
        logger.warning("Fetching all ID-to-row mappings from Pinecone can be inefficient and is not directly supported. This method may be removed or changed.")
        raise NotImplementedError("Fetching all ID-to-row mappings from Pinecone is not efficiently supported.")

    def get_all_texts(self) -> List[str]:
        # Implementation will follow
        logger.warning("Fetching all texts from Pinecone can be inefficient and is not directly supported. This method may be removed or changed.")
        raise NotImplementedError("Fetching all texts from Pinecone is not efficiently supported.")

    def get_embedding(self, hash_id: str) -> Optional[np.ndarray]:
        if not hash_id:
            return None
        try:
            fetch_response = self.index.fetch(ids=[hash_id], namespace=self.namespace)
            if fetch_response and fetch_response.vectors and hash_id in fetch_response.vectors:
                # Convert list of floats to np.ndarray
                return np.array(fetch_response.vectors[hash_id].values, dtype=np.float32)
            else:
                logger.info(f"Embedding for hash_id '{hash_id}' not found in Pinecone namespace '{self.namespace}'.")
                return None
        except Exception as e:
            logger.error(f"Error fetching embedding for hash_id '{hash_id}' from Pinecone: {e}")
            return None

    def get_embeddings(self, hash_ids: List[str]) -> List[Optional[np.ndarray]]:
        if not hash_ids:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(hash_ids)
        # Map hash_id to its original index to restore order later
        hash_id_to_original_index = {h_id: i for i, h_id in enumerate(hash_ids)}

        try:
            # Batching fetch requests if hash_ids list is very long
            for i in range(0, len(hash_ids), self.batch_size): # Using self.batch_size for fetch batching
                batch_ids_to_fetch = hash_ids[i:i + self.batch_size]
                if not batch_ids_to_fetch: continue
                
                fetch_response = self.index.fetch(ids=batch_ids_to_fetch, namespace=self.namespace)
                if fetch_response and fetch_response.vectors:
                    for h_id, vector_data in fetch_response.vectors.items():
                        original_index = hash_id_to_original_index.get(h_id)
                        if original_index is not None: # Should always be found
                            results[original_index] = np.array(vector_data.values, dtype=np.float32)
                        else: # Should not happen if logic is correct
                            logger.warning(f"Fetched hash_id '{h_id}' not found in original request list during get_embeddings.")
                # IDs not in fetch_response.vectors will remain None in results, which is correct.
        
        except Exception as e:
            logger.error(f"Error fetching embeddings for hash_ids from Pinecone: {e}")
            # Results will contain Nones for all entries if an exception occurs.
            # Or, more precisely, for entries that couldn't be fetched before the error.
        
        return results

    def delete(self, hash_ids_to_delete: List[str]):
        if not hash_ids_to_delete:
            logger.info("No hash_ids provided for deletion.")
            return

        try:
            # Pinecone delete can take a list of IDs.
            # It's generally efficient, but check Pinecone docs for limits on list size if any.
            # For very large lists, batching might be needed, but typically not for delete.
            self.index.delete(ids=hash_ids_to_delete, namespace=self.namespace)
            logger.info(f"Successfully initiated deletion for {len(hash_ids_to_delete)} IDs from Pinecone index '{self.index_name}' namespace '{self.namespace}'.")
            # Note: Pinecone delete is eventually consistent. Confirmation of deletion might require a subsequent check if needed.
        except Exception as e:
            logger.error(f"Error deleting vectors from Pinecone: {e}")
            # Potentially re-raise or handle more gracefully
            raise
        pass

    def get_hash_id_for_text(self, text: str) -> Optional[str]:
        """
        Retrieves the hash_id for a given text by querying Pinecone metadata.
        Note: This operation can be inefficient in Pinecone if the 'content' metadata field
        is not indexed appropriately, as it may involve scanning.
        Pinecone is primarily optimized for vector similarity search, not exact metadata matches across many records.
        """
        if not text:
            return None

        # Pinecone does not directly support querying for exact text match in metadata efficiently across the entire dataset
        # without specific metadata indexing strategies that might not be universally available or performant.
        # The most straightforward way, though potentially inefficient, is to generate the expected hash_id
        # and then fetch by ID to see if it exists and if its metadata (content) matches.
        # However, the task is to find the hash_id *given the text*.

        # Approach 1: Generate hash_id and fetch to verify (assumes hash_id generation is consistent)
        # This is more about verifying an existing text-hash pair than discovering a hash from text alone
        # if the text itself isn't part of the ID.
        # Here, our hash_id *is* derived from the text. So, we can generate it and check.
        
        expected_hash_id = compute_mdhash_id(text) # Consistent with how IDs are generated in get_missing_string_hash_ids

        try:
            fetch_response = self.index.fetch(ids=[expected_hash_id], namespace=self.namespace)
            if fetch_response and fetch_response.vectors and expected_hash_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[expected_hash_id]
                # Verify that the content in metadata actually matches, just in case of hash collisions (unlikely for MD5)
                # or if the ID generation scheme changes.
                if vector_data.metadata and vector_data.metadata.get("content") == text:
                    return expected_hash_id
                else:
                    # This case (hash_id exists but content doesn't match) should ideally not happen
                    # if hash_id is a hash of the content.
                    logger.warning(f"Found vector for hash_id '{expected_hash_id}' but content in metadata does not match for Pinecone.")
                    return None 
            else:
                # logger.info(f"Hash ID for text not found in Pinecone: '{expected_hash_id}'")
                return None
        except Exception as e:
            logger.error(f"Error fetching hash_id for text from Pinecone: {e}")
            return None

        # Approach 2: Using query with a dummy vector and metadata filter (if metadata indexing is enabled and suitable)
        # This is generally not recommended for this exact use case due to inefficiency.
        # Example (pseudo-code, actual filter syntax might vary or not be efficient):
        # try:
        #     # Create a dummy query vector (e.g., zeros)
        #     # This is not ideal as it still performs a vector search.
        #     if not self.dimension:
        #         logger.error("Cannot query by text: dimension is unknown for PineconeVectorStore.")
        #         return None
        #     dummy_vector = [0.0] * self.dimension 
        #     query_response = self.index.query(
        #         vector=dummy_vector,
        #         filter={"content": {"$eq": text}}, # Pinecone filter syntax
        #         top_k=1,
        #         namespace=self.namespace,
        #         include_metadata=True 
        #     )
        #     if query_response and query_response.matches:
        #         # Verify content again, as filter might not be perfect depending on indexing
        #         match = query_response.matches[0]
        #         if match.metadata and match.metadata.get("content") == text:
        #             return match.id 
        #         else: # Hash ID in metadata could be used if stored: return match.metadata.get("hash_id")
        #             logger.warning(f"Query returned a match for text, but content verification failed or hash_id missing in metadata.")
        #             return None
        #     return None
        # except Exception as e:
        #     logger.error(f"Error querying by text metadata from Pinecone: {e}")
        #     return None

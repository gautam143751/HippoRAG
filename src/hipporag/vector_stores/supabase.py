import os
import logging
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd # Though not directly used in all methods, good for data manipulation if needed
from supabase import create_client, ClientOptions # Corrected import if needed

from ..base import VectorStore
from ..utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)

class SupabaseVectorStore(VectorStore):
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        embedding_model: Any,
        namespace: str, # namespace is used for hash_id prefixing primarily
        table_name: str = "embeddings",
        batch_size: int = 32,
        client_options: Optional[ClientOptions] = None, # For advanced Supabase client config
    ):
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key must be provided.")

        self.client = create_client(supabase_url, supabase_key, options=client_options if client_options else ClientOptions())
        self.embedding_model = embedding_model
        self.namespace = namespace # Used for prefixing hash_ids
        self.table_name = table_name
        self.batch_size = batch_size
        
        logger.info(f"SupabaseVectorStore initialized for table '{self.table_name}' and namespace '{self.namespace}'")

    def _load_data(self):
        # This method could be used to ensure the table exists and has the correct schema,
        # or to perform any initial setup required for the Supabase table.
        # For now, we'll log a message.
        logger.info(f"[_load_data] Supabase handles data persistence. Table '{self.table_name}' is assumed to exist or be managed externally.")
        # Example: Check if table exists (pseudo-code, actual implementation depends on Supabase features/permissions)
        # try:
        #     self.client.table(self.table_name).select("hash_id", count="exact").limit(0).execute()
        #     logger.info(f"Table '{self.table_name}' found.")
        # except Exception as e:
        #     logger.error(f"Error checking table '{self.table_name}': {e}. It might not exist or schema needs verification.")
        pass

    def _save_data(self):
        # Direct saving to a file isn't needed as Supabase handles persistence.
        logger.info(f"[_save_data] Supabase handles data persistence directly. No file saving operation performed.")
        pass

    def _upsert(self, hash_ids: List[str], texts: List[str], embeddings: List[Any]):
        if not hash_ids:
            return

        records_to_upsert = []
        for h_id, text, emb in zip(hash_ids, texts, embeddings):
            # Convert numpy array embedding to list of floats if necessary
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb) # Ensure it's a list

            records_to_upsert.append({
                "hash_id": h_id,
                "content": text,
                "embedding": emb_list,
                "namespace": self.namespace # Assuming a namespace column for filtering
            })
        
        try:
            # Supabase client's `upsert` method is suitable here.
            # It will insert new rows or update existing ones if hash_id matches.
            response = self.client.table(self.table_name).upsert(records_to_upsert).execute()
            if response.data:
                 logger.info(f"Successfully upserted {len(response.data)} records into '{self.table_name}'.")
            else: # Check for errors if possible, response structure varies.
                # Supabase-py v2 might not have detailed error info directly in response.data for upsert
                # It might raise an APIError for issues.
                # If response.error exists (older versions or certain conditions)
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error upserting records into '{self.table_name}': {response.error}")
                    # raise Exception(f"Supabase upsert error: {response.error}")
                else: # If no data and no explicit error, it's an ambiguous success or partial failure.
                      # This part might need adjustment based on exact Supabase client behavior on errors.
                    logger.warning(f"Upsert into '{self.table_name}' completed, but response data is empty. Verify success. Records attempted: {len(records_to_upsert)}")

        except Exception as e: # Catching generic Exception, specific Supabase/API errors preferred
            logger.error(f"Exception during upsert to Supabase table '{self.table_name}': {e}")
            # Depending on desired behavior, you might want to re-raise the exception
            # raise e
        pass

    def insert_strings(self, texts: List[str]):
        if not texts:
            return

        # 1. Generate hash_id for each text
        # The namespace is part of the hash_id generation itself.
        prospective_hash_ids = [compute_mdhash_id(text, prefix=f"{self.namespace}-") for text in texts]
        
        # Create a temporary mapping from prospective hash_id to original text
        hash_id_to_text_map = {h_id: text for h_id, text in zip(prospective_hash_ids, texts)}

        # 2. Check which texts/hash_ids are missing
        # get_missing_string_hash_ids should ideally just return the list of hash_ids that are missing
        # or a structure from which we can easily derive this.
        # For now, assuming get_missing_string_hash_ids queries Supabase and returns a dict for missing items.
        # The base class definition for get_missing_string_hash_ids is `(self, texts: List[str]) -> Dict[str, Dict[str, Any]]`
        # which means it expects texts, generates hash_ids internally, queries, and returns a dict of missing hash_id -> {hash_id, content}
        
        missing_info: Dict[str, Dict[str, Any]] = self.get_missing_string_hash_ids(texts) # This will internally re-calculate hash_ids

        if not missing_info:
            logger.info("No new strings to insert. All provided texts already exist.")
            return

        new_hash_ids_to_insert = list(missing_info.keys())
        new_texts_to_insert = [missing_info[h_id]["content"] for h_id in new_hash_ids_to_insert]

        if not new_texts_to_insert: # Should not happen if missing_info is not empty, but as a safeguard
            logger.info("No new strings to insert based on missing info check.")
            return

        logger.info(f"Found {len(new_texts_to_insert)} new strings to insert.")

        # 3. Generate embeddings for the new texts
        new_embeddings = self.embedding_model.batch_encode(new_texts_to_insert, batch_size=self.batch_size)

        # 4. Call _upsert to store the new hash_ids, texts, and embeddings
        self._upsert(new_hash_ids_to_insert, new_texts_to_insert, new_embeddings)
        
        # 5. Return None (implicitly, as per void return type)
        return

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        if not texts:
            return {}

        # 1. Generate hash_id for each text
        prospective_hash_id_to_text_map: Dict[str, str] = {
            compute_mdhash_id(text, prefix=f"{self.namespace}-"): text for text in texts
        }
        
        if not prospective_hash_id_to_text_map: # Should not happen if texts is not empty
            return {}

        prospective_hash_ids = list(prospective_hash_id_to_text_map.keys())

        # 2. Query Supabase table to find which hash_ids already exist
        # We must also filter by namespace if our table stores data for multiple namespaces.
        # The hash_id itself is already namespaced, but if a 'namespace' column exists, filter by it too.
        try:
            response = self.client.table(self.table_name)\
                .select("hash_id")\
                .in_("hash_id", prospective_hash_ids)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                existing_hash_ids = {item['hash_id'] for item in response.data}
            else:
                existing_hash_ids = set()
                if hasattr(response, 'error') and response.error:
                     logger.error(f"Error fetching existing hash_ids from '{self.table_name}': {response.error}")
                     # Depending on desired behavior, might raise or return all as missing.
                     # For now, assume if error, all are potentially missing (or handle error more gracefully)

        except Exception as e:
            logger.error(f"Exception fetching existing hash_ids from Supabase table '{self.table_name}': {e}")
            # If the query fails, we might assume all texts are missing, or re-raise.
            # For robustness, let's assume they are all missing to avoid data loss if the DB is temporarily down.
            # However, this could lead to re-embedding if not careful.
            # A safer approach might be to re-raise or return an empty dict indicating failure.
            # For now, returning all as potentially missing if query fails.
            existing_hash_ids = set() # Or handle more specific error reporting

        # 3. Determine missing hash_ids and prepare the result
        missing_texts_info: Dict[str, Dict[str, Any]] = {}
        for h_id, text_content in prospective_hash_id_to_text_map.items():
            if h_id not in existing_hash_ids:
                missing_texts_info[h_id] = {"hash_id": h_id, "content": text_content}
        
        return missing_texts_info

    def get_row(self, hash_id: str) -> Optional[Dict[str, Any]]:
        if not hash_id:
            return None
        try:
            response = self.client.table(self.table_name)\
                .select("hash_id, content")\
                .eq("hash_id", hash_id)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .limit(1)\
                .execute()
            
            if response.data:
                # Supabase returns a list, get the first item
                row_data = response.data[0]
                return {"hash_id": row_data['hash_id'], "content": row_data['content']}
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching row for hash_id '{hash_id}' from '{self.table_name}': {response.error}")
                return None # Not found or error occurred
        except Exception as e:
            logger.error(f"Exception fetching row for hash_id '{hash_id}' from Supabase table '{self.table_name}': {e}")
            return None

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not hash_ids:
            return {}
        
        rows_dict: Dict[str, Dict[str, Any]] = {}
        # Batching requests if hash_ids list is very long might be necessary,
        # but Supabase's `in_` filter is generally efficient for a reasonable number of IDs.
        # Let's assume the list size is manageable for a single query.
        try:
            response = self.client.table(self.table_name)\
                .select("hash_id, content")\
                .in_("hash_id", hash_ids)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                for item in response.data:
                    rows_dict[item['hash_id']] = {"hash_id": item['hash_id'], "content": item['content']}
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching rows for hash_ids from '{self.table_name}': {response.error}")
        
        except Exception as e:
            logger.error(f"Exception fetching rows for hash_ids from Supabase table '{self.table_name}': {e}")
            # Depending on requirements, might return partially fetched data or an empty dict on error.
            # For now, returns what was fetched before an error, or empty if error at start.

        return rows_dict

    def get_all_ids(self) -> List[str]:
        all_hash_ids: List[str] = []
        try:
            # Paginate through results if necessary, Supabase might limit query results.
            # For now, assuming a manageable number of rows or that client handles pagination.
            # A more robust solution would implement pagination.
            response = self.client.table(self.table_name)\
                .select("hash_id")\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                for item in response.data:
                    all_hash_ids.append(item['hash_id'])
            else:
                if hasattr(response, 'error') and response.error:
                     logger.error(f"Error fetching all hash_ids from '{self.table_name}': {response.error}")
        
        except Exception as e:
            logger.error(f"Exception fetching all hash_ids from Supabase table '{self.table_name}': {e}")
            # Return what has been fetched so far, or empty list if error at start.

        return all_hash_ids

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        all_rows_map: Dict[str, Dict[str, Any]] = {}
        try:
            # Paginate through results if necessary
            # A more robust solution would implement pagination.
            response = self.client.table(self.table_name)\
                .select("hash_id, content")\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                for item in response.data:
                    all_rows_map[item['hash_id']] = {"hash_id": item['hash_id'], "content": item['content']}
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching all rows from '{self.table_name}': {response.error}")
        
        except Exception as e:
            logger.error(f"Exception fetching all rows from Supabase table '{self.table_name}': {e}")

        return all_rows_map

    def get_all_texts(self) -> List[str]:
        all_texts_list: List[str] = []
        try:
            # Paginate through results if necessary
            response = self.client.table(self.table_name)\
                .select("content")\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                for item in response.data:
                    all_texts_list.append(item['content'])
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching all texts from '{self.table_name}': {response.error}")

        except Exception as e:
            logger.error(f"Exception fetching all texts from Supabase table '{self.table_name}': {e}")
            
        return all_texts_list

    def get_embedding(self, hash_id: str) -> Optional[np.ndarray]:
        if not hash_id:
            return None
        try:
            response = self.client.table(self.table_name)\
                .select("embedding")\
                .eq("hash_id", hash_id)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .limit(1)\
                .execute()
            
            if response.data:
                embedding_list = response.data[0].get('embedding')
                if embedding_list:
                    return np.array(embedding_list, dtype=np.float32)
                else:
                    logger.warning(f"Embedding data not found for hash_id '{hash_id}' in response.")
                    return None
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching embedding for hash_id '{hash_id}': {response.error}")
                return None # Not found or error
        except Exception as e:
            logger.error(f"Exception fetching embedding for hash_id '{hash_id}': {e}")
            return None

    def get_embeddings(self, hash_ids: List[str]) -> List[Optional[np.ndarray]]:
        if not hash_ids:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(hash_ids)
        hash_id_to_index_map = {hash_id: i for i, hash_id in enumerate(hash_ids)}

        try:
            response = self.client.table(self.table_name)\
                .select("hash_id, embedding")\
                .in_("hash_id", hash_ids)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            if response.data:
                for item in response.data:
                    h_id = item.get('hash_id')
                    embedding_list = item.get('embedding')
                    if h_id and embedding_list:
                        idx = hash_id_to_index_map.get(h_id)
                        if idx is not None: # Should always be found if h_id is from response
                            results[idx] = np.array(embedding_list, dtype=np.float32)
                    elif h_id:
                        logger.warning(f"Embedding data not found for hash_id '{h_id}' in get_embeddings response.")
            else:
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error fetching embeddings for hash_ids: {response.error}")
        
        except Exception as e:
            logger.error(f"Exception fetching embeddings for hash_ids: {e}")
            # Results will contain Nones for all entries if an exception occurs mid-process or at start.
        
        return results

    def delete(self, hash_ids_to_delete: List[str]):
        if not hash_ids_to_delete:
            logger.info("No hash_ids provided for deletion.")
            return

        try:
            # Delete rows that match any of the hash_ids in the list and the namespace
            response = self.client.table(self.table_name)\
                .delete()\
                .in_("hash_id", hash_ids_to_delete)\
                .eq("namespace", self.namespace) # Filter by namespace column
                .execute()

            # Supabase delete usually returns data in response.data for the deleted rows.
            # The number of items in response.data can be used to confirm deletions.
            if response.data:
                logger.info(f"Successfully deleted {len(response.data)} records from '{self.table_name}'.")
            else:
                # If no data, it could mean no rows matched, or an error occurred not raising an exception.
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Error deleting records from '{self.table_name}': {response.error}")
                else:
                    # This case might mean no records matched the criteria.
                    logger.info(f"No records found matching the provided hash_ids for deletion in namespace '{self.namespace}'. Or an unconfirmed error occurred.")
        
        except Exception as e:
            logger.error(f"Exception during deletion from Supabase table '{self.table_name}': {e}")
            # Depending on desired behavior, you might want to re-raise the exception
            # raise e
        pass

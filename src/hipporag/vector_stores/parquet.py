import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from copy import deepcopy
from typing import List, Any, Dict, Optional

from ..base import VectorStore
from ..utils.misc_utils import compute_mdhash_id # Changed back

class ParquetVectorStore(VectorStore):
    def __init__(
        self,
        db_directory: str, # Changed from db_filename to db_directory
        embedding_model: Any, # Replace Any with the actual type of EmbeddingModel
        batch_size: int = 32,
        namespace: Optional[str] = "default_namespace", # Added default for filename construction
    ):
        # self.db_filename = db_filename # Old line
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_directory): # Use db_directory here
            logging.info(f"Creating working directory: {db_directory}")
            os.makedirs(db_directory, exist_ok=True)
        
        # Construct the actual Parquet filename
        self.db_filename = os.path.join(
            db_directory, f"vdb_{self.namespace}.parquet"
        )
        logging.info(f"Parquet database file set to: {self.db_filename}")

        self.df: Optional[pd.DataFrame] = None
        self.text_to_hash_id: Dict[str, str] = {}
        self._load_data()

    def insert_strings(self, texts: List[str]):
        if not texts:
            return {}

        # Compute hash IDs for all input texts
        # The namespace is used here to ensure hash_ids are unique if the same text is used in different namespaces.
        # If self.namespace is None, it behaves as before.
        current_hash_ids = [compute_mdhash_id(text, prefix=f"{self.namespace}-" if self.namespace else "") for text in texts] # Changed back
        
        # Identify which texts are new
        new_texts_to_insert = []
        new_hash_ids_to_insert = []

        for i, text in enumerate(texts):
            h_id = current_hash_ids[i]
            if h_id not in self.hash_id_to_idx: # Check against loaded/existing hash_ids
                new_texts_to_insert.append(text)
                new_hash_ids_to_insert.append(h_id)

        logging.info(
            f"Inserting {len(new_hash_ids_to_insert)} new records. "
            f"{len(texts) - len(new_hash_ids_to_insert)} records already exist."
        )

        if not new_texts_to_insert:
            # All records already exist. According to the requirement, insert_strings should return None.
            logging.info("All records already exist. No new records to insert.")
            return

        # Compute embeddings for the new texts
        # Assuming self.embedding_model.batch_encode exists and works as in EmbeddingStore
        new_embeddings = self.embedding_model.batch_encode(new_texts_to_insert, batch_size=self.batch_size)

        # Upsert the new data
        self._upsert(new_hash_ids_to_insert, new_texts_to_insert, new_embeddings)
        
        # Return rows for all originally requested texts (both new and existing)
        # This requires that hash_id_to_row is updated by _upsert
        # result_rows = {h_id: self.hash_id_to_row[h_id] for h_id in current_hash_ids if h_id in self.hash_id_to_row}
        # return result_rows
        # The original EmbeddingStore.insert_strings didn't explicitly return rows, let's stick to that for now or clarify.
        # The abstract method does not specify a return type, so `None` is acceptable.
        # Returning None.
        return

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, Dict[str, Any]]: # Changed return type
        if not texts:
            return {} # Return empty dict

        missing_texts_info: Dict[str, Dict[str, Any]] = {} # Changed variable name and type
        for text in texts:
            # Consistent hash ID generation with insert_strings
            h_id = compute_mdhash_id(text, prefix=f"{self.namespace}-" if self.namespace else "") # Name already changed
            if h_id not in self.hash_id_to_idx: # Check if hash_id is missing
                missing_texts_info[h_id] = {"hash_id": h_id, "content": text} # Populate dict
        
        return missing_texts_info # Return the dictionary

    def get_row(self, hash_id: str) -> Optional[Dict[str, Any]]:
        return deepcopy(self.hash_id_to_row.get(hash_id))

    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not hash_ids:
            return {}
        # results = {id_str: deepcopy(self.hash_id_to_row.get(id_str)) for id_str in hash_ids if id_str in self.hash_id_to_row}
        # Filter out None results if a hash_id is not found, or ensure get_row handles returning a copy.
        # The current get_row returns a deepcopy, so this is fine.
        results = {}
        for id_str in hash_ids:
            row = self.get_row(id_str) # This already deepcopies
            if row: # Only add if the row exists
                results[id_str] = row
        return results

    def get_all_ids(self) -> List[str]:
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self) -> List[str]: # Changed from Set to List to match self.texts
        return deepcopy(self.texts) # self.texts should already store all unique texts if managed correctly

    def get_embedding(self, hash_id: str, dtype=np.float32) -> Optional[np.ndarray]:
        if hash_id not in self.hash_id_to_idx:
            return None
        idx = self.hash_id_to_idx[hash_id]
        embedding = self.embeddings[idx]
        # Ensure embedding is a numpy array and convert its dtype.
        # The stored embeddings might be lists or arrays of various types.
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        return embedding.astype(dtype)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> List[Optional[np.ndarray]]:
        if not hash_ids:
            return []
        
        results: List[Optional[np.ndarray]] = []
        for h_id in hash_ids:
            embedding = self.get_embedding(h_id, dtype=dtype) # Reuses the logic in get_embedding
            results.append(embedding) # Appends None if embedding not found
        return results
        # Original EmbeddingStore logic:
        # indices = np.array([self.hash_id_to_idx[h] for h in hash_ids if h in self.hash_id_to_idx], dtype=np.intp)
        # if len(indices) == 0:
        #     return []
        # embeddings_list = [self.embeddings[idx] for idx in indices]
        # # Ensure all elements are numpy arrays and convert dtype
        # processed_embeddings = []
        # for emb in embeddings_list:
        #     if not isinstance(emb, np.ndarray):
        #         emb = np.array(emb)
        #     processed_embeddings.append(emb.astype(dtype))
        # return processed_embeddings
        # The new approach is slightly different: it returns a list that includes None for missing hash_ids,
        # maintaining the order and length of the input hash_ids list. This might be more user-friendly.

    def delete(self, hash_ids_to_delete: List[str]):
        if not hash_ids_to_delete:
            logging.info("No hash_ids provided for deletion.")
            return

        # Identify valid indices to delete, and sort them in descending order
        # to avoid issues with list index changes during pop operations.
        indices_to_delete = sorted(
            [self.hash_id_to_idx[h_id] for h_id in hash_ids_to_delete if h_id in self.hash_id_to_idx],
            reverse=True
        )

        if not indices_to_delete:
            logging.info("None of the provided hash_ids found for deletion.")
            return

        num_deleted = len(indices_to_delete)

        for idx in indices_to_delete:
            # Remove from primary lists
            deleted_hash_id = self.hash_ids.pop(idx)
            deleted_text = self.texts.pop(idx)
            self.embeddings.pop(idx) # Assuming self.embeddings is always parallel to self.hash_ids and self.texts

            # Remove from mappings
            del self.hash_id_to_idx[deleted_hash_id]
            del self.hash_id_to_row[deleted_hash_id]
            if deleted_text in self.text_to_hash_id and self.text_to_hash_id[deleted_text] == deleted_hash_id:
                 # Only delete if this text instance specifically maps to the deleted hash_id
                 # This handles cases where different texts might produce the same hash if not careful with hashing,
                 # or if a text somehow got associated with multiple hash_ids (though current logic tries to avoid this).
                del self.text_to_hash_id[deleted_text]
        
        # Rebuild hash_id_to_idx after deletions to ensure correctness
        self.hash_id_to_idx = {h_id: i for i, h_id in enumerate(self.hash_ids)}

        logging.info(f"Deleted {num_deleted} records. Saving updated data.")
        self._save_data()


    def _load_data(self):
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids: List[str] = df["hash_id"].values.tolist()
            self.texts: List[str] = df["content"].values.tolist()
            self.embeddings: List[Any] = df["embedding"].values.tolist() # Or List[np.ndarray]
            
            self.hash_id_to_idx: Dict[str, int] = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row: Dict[str, Dict[str, Any]] = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            # self.hash_id_to_text: Dict[str, str] = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)} # Redundant with hash_id_to_row
            self.text_to_hash_id: Dict[str, str] = {t: h for h, t in zip(self.hash_ids, self.texts)} # Ensure texts are unique for this to be reliable
            
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logging.info(f"Loaded {len(self.hash_ids)} records from {self.db_filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row, self.text_to_hash_id = {}, {}, {}

    def _save_data(self):
        # Before saving, ensure internal lists are consistent.
        # This might be more robust if we rebuild from hash_id_to_row and hash_id_to_idx if they are the source of truth.
        # For now, assuming self.hash_ids, self.texts, self.embeddings are the primary lists.
        
        # Rebuild texts and embeddings from hash_id_to_row and hash_id_to_idx to ensure order
        ordered_texts = []
        ordered_embeddings = []
        
        # Create a temporary mapping from hash_id to embedding, as embeddings are not in hash_id_to_row
        temp_hash_id_to_embedding = {self.hash_ids[i]: self.embeddings[i] for i in range(len(self.hash_ids))}

        for hash_id in self.hash_ids: # Iterate in the current order of hash_ids
            ordered_texts.append(self.hash_id_to_row[hash_id]['content'])
            ordered_embeddings.append(temp_hash_id_to_embedding[hash_id])

        df_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": ordered_texts, # Use the reordered texts
            "embedding": ordered_embeddings # Use the reordered embeddings
        })
        df_to_save.to_parquet(self.db_filename, index=False)
        
        # Refresh auxiliary mappings after saving, ensuring consistency
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t}
            for h, t in zip(self.hash_ids, self.texts) # self.texts should align with self.hash_ids
        }
        self.text_to_hash_id = {t: h for h, t in zip(self.texts, self.hash_ids)} # self.texts should align with self.hash_ids
        logging.info(f"Saved {len(self.hash_ids)} records to {self.db_filename}")

    def _upsert(self, hash_ids: List[str], texts: List[str], embeddings: List[Any]):
        # Add new data to internal lists
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        # Update mappings for the new data
        for i in range(len(hash_ids)):
            h_id = hash_ids[i]
            text = texts[i]
            # embedding = embeddings[i] # Embedding not stored in hash_id_to_row

            new_idx = len(self.hash_ids) - len(hash_ids) + i # Calculate new index
            self.hash_id_to_idx[h_id] = new_idx
            self.hash_id_to_row[h_id] = {"hash_id": h_id, "content": text}
            self.text_to_hash_id[text] = h_id
            
        logging.info(f"Upserting {len(hash_ids)} new records. Total records will be {len(self.hash_ids)}.")
        self._save_data()

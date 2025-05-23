from abc import ABC, abstractmethod
from typing import List, Any

class VectorStore(ABC):
    @abstractmethod
    def insert_strings(self, texts: List[str]):
        pass

    @abstractmethod
    def get_missing_string_hash_ids(self, texts: List[str]):
        pass

    @abstractmethod
    def get_row(self, hash_id: str):
        pass

    @abstractmethod
    def get_rows(self, hash_ids: List[str]):
        pass

    @abstractmethod
    def get_all_ids(self):
        pass

    @abstractmethod
    def get_all_id_to_rows(self):
        pass

    @abstractmethod
    def get_all_texts(self):
        pass

    @abstractmethod
    def get_embedding(self, hash_id: str):
        pass

    @abstractmethod
    def get_embeddings(self, hash_ids: List[str]):
        pass

    @abstractmethod
    def delete(self, hash_ids: List[str]):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _save_data(self):
        pass

    @abstractmethod
    def _upsert(self, hash_ids: List[str], texts: List[str], embeddings: List[Any]):
        pass

    @abstractmethod
    def get_hash_id_for_text(self, text: str) -> Optional[str]: # Added Optional
        """Returns the hash_id for a given text if it exists in the store."""
        pass

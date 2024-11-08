from typing import Optional
from llama_index.vector_stores.qdrant import (
    QdrantVectorStore as LIQdrantVectorStore,
)
from qdrant_client import QdrantClient
from rag.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def get_llama_index_store(self):
        return LIQdrantVectorStore(
            collection_name=self.collection_name, client=self.client
        )

    async def is_index_live(self):
        return self.client.collection_exists(self.collection_name)

    async def delete_index(self) -> None:
        return self.client.delete_collection(self.collection_name)

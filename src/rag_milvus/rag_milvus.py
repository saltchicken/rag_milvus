from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

class Rag():
    """Retrieval Augmented Generation (RAG) implementation using Milvus vector database.

    This class provides functionality to store, embed, and retrieve knowledge using
    semantic search capabilities. It uses SentenceTransformer for text embeddings
    and Milvus for vector similarity search.

    Attributes:
        model: SentenceTransformer instance for text embedding
        connection: Milvus database connection

    Args:
        host (str): Milvus server host address. Defaults to "localhost"
        port (str): Milvus server port. Defaults to "19530"
    """
    def __init__(self, host="localhost", port="19530"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.connection = connections.connect(host=host, port=port)
        self.create_necessary_collections()

    def create_necessary_collections(self, replace=False):
        collection_name = "knowledge_base"
        if utility.has_collection(collection_name):
            if replace:
                utility.drop_collection(collection_name)
            else:
                print(f"Knowledge base already exists. Skipping creation.")
                return
        knowledge_base_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="knowledge", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="knowledge_embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        knowledge_base_collection = self.create_collection(collection_name, knowledge_base_fields)
        index_params_knowledge_embedding = {
            "metric_type": "L2",
            "index_type": "GPU_IVF_FLAT",
            "params": {"nlist": 100},
        }
        knowledge_base_collection.create_index(field_name="knowledge_embedding", index_params=index_params_knowledge_embedding)

    def create_collection(self, collection_name, fields=None):

        # TODO: Add a description
        schema = CollectionSchema(fields=fields, description="Enter description here")
        collection = Collection(name=collection_name, schema=schema)
        return collection

    def write_knowledge(self, knowledge, collection_name="knowledge_base"):
        collection = Collection(collection_name)
        knowledge_embedding = self.model.encode(knowledge)

        collection.insert({
            "knowledge": knowledge,
            "knowledge_embedding": knowledge_embedding,
            "timestamp": int(np.datetime64("now").astype(int))
        })

    def search_knowledge(self, query, recall_threshold=0.01):
        collection = Collection("knowledge_base")
        collection.load()
        query_embedding = self.model.encode(query)

        results = collection.search(
            data=[query_embedding],
            anns_field="knowledge_embedding",
            param={"metric_type": "L2", "params": {"nprobe": 100}},
            limit=10,
            output_fields=["knowledge"],
        )

        results = results[0]
        print(results)

        results = [hit.entity.get('knowledge') for hit in results if hit.score < recall_threshold]

        return results

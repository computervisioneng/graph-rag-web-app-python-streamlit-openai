from typing import Dict, Any
import os

from PIL import Image
from img2vec_pytorch import Img2Vec
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import register_embedding_function
from graphrag.config.defaults import (
	DEFAULT_COMPLETION_MODEL,
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_MODEL_PROVIDER,
	DEFAULT_EMBEDDING_MODEL_AUTH_TYPE,
	DEFAULT_COMPLETION_MODEL_AUTH_TYPE
)
from graphrag.config.models.graph_rag_config import GraphRagConfig


DB_PATH = "./db"
DATA_PATH = "./data"


def get_config(img_category):
    config = GraphRagConfig(embedding_models={"default_embedding_model": {"model": DEFAULT_EMBEDDING_MODEL,
                                                                          "model_provider": DEFAULT_MODEL_PROVIDER,
                                                                          "auth_method": DEFAULT_EMBEDDING_MODEL_AUTH_TYPE,
                                                                          "api_key": os.getenv("GRAPHRAG_API_KEY")}},
                            completion_models={"default_completion_model": {"model": DEFAULT_COMPLETION_MODEL,
                                                                            "model_provider": DEFAULT_MODEL_PROVIDER,
                                                                            "auth_method": DEFAULT_COMPLETION_MODEL_AUTH_TYPE,
                                                                            "api_key": os.getenv("GRAPHRAG_API_KEY")}})

    #  update completion and embedding models

    #  update input, output directories
    config.input_storage.base_dir = f"/home/phillip/Documents/current_tutorial/131_graph_rag_python/code/data/{img_category}"
    config.output_storage.base_dir = f"/home/phillip/Documents/current_tutorial/131_graph_rag_python/code/output/{img_category}"
    config.update_output_storage.base_dir = f"/home/phillip/Documents/current_tutorial/131_graph_rag_python/code/update_output/{img_category}"
    config.vector_store.db_uri = f"/home/phillip/Documents/current_tutorial/131_graph_rag_python/code/output/{img_category}/lancedb"

    return config


@register_embedding_function
class ImageEmbeddings(EmbeddingFunction):

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        embeddings = self._get_imgs_embeddings(input)
        return embeddings

    def _get_imgs_embeddings(self, input):
        return [self.model.get_vec(Image.fromarray(img)) for img in input]

    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddings(config['model'])

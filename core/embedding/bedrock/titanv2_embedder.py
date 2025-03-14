
from typing import Dict, List
import logging

from core.embedding import EmbedderFactory
from baseclasses.base_classes import BaseEmbedder
from . import BedrockEmbedder


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TitanV2Embedder(BedrockEmbedder):
    def prepare_payload(self, text: str, dimensions: int, normalize: bool) -> Dict:
        return {"inputText": text, "dimensions": dimensions, "normalize": normalize}

    def extract_embedding(self, response: Dict) -> List[float]:
        return response["embedding"]

EmbedderFactory.register_embedder("bedrock", "amazon.titan-embed-text-v2:0", TitanV2Embedder)
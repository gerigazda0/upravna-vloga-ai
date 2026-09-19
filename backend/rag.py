"""
Modul za RAG iskanje.
Uporablja že izračunane OpenAI embeddinge iz lokalne Chroma zbirke.
"""
import os
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "dgd_regulations"


class RAGRetriever:
    def __init__(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY ni nastavljen")
        self._openai = OpenAI(api_key=key)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Če ingest.py še ni bil pognan, tu vrže InvalidCollectionException
        self._col = client.get_collection(name=COLLECTION_NAME, embedding_function=None)

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return resp.data[0].embedding

    def retrieve(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        total = self._col.count()
        if total == 0:
            return []
        n = min(k, total)
        q_emb = self._embed(query[:2000])
        results = self._col.query(query_embeddings=[q_emb], n_results=n)
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            chunks.append({
                "text":    doc,
                "source":  meta.get("source", "neznano"),
                "article": meta.get("article", ""),
            })
        return chunks

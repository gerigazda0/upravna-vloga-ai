#!/usr/bin/env python3
"""
Naloži besedila predpisov iz knowledge_base/ v lokalno Chroma vektorsko bazo.

Zaženeš (iz mape backend/):
    python ingest.py

Ob vsakem zagonu zbirko pobriše in jo zgradi na novo.
"""
import os
import re
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_PATH        = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME    = "dgd_regulations"

CHUNK_TARGET  = 700   # ciljna velikost delca v znakih
CHUNK_OVERLAP = 100   # koliko se sosednja delca prekrivata


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Reže po znakih, rez pa poravna na konec stavka in doda prekrivanje.
    Vsak delec ima zraven metapodatke: ime datoteke in najbližji naslov člena.
    """
    article_re = re.compile(r'(\d+\.\s*člen\b)', re.IGNORECASE)
    chunks: list[dict] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + CHUNK_TARGET, len(text))

        # Poravnaj na naravno mejo, da ne sekamo sredi stavka
        if end < len(text):
            for sep in ("\n\n", ".\n", ". ", "\n"):
                idx = text.rfind(sep, start + CHUNK_TARGET // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        body = text[start:end].strip()
        if body:
            m = article_re.search(body)
            chunks.append({
                "text": body,
                "metadata": {
                    "source":   source,
                    "article":  m.group(0) if m else "",
                    "chunk_id": chunk_id,
                },
            })
            chunk_id += 1

        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Seznam besedil pretvori v vektorje s text-embedding-3-small."""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("NAPAKA: OPENAI_API_KEY ni nastavljen v .env", file=sys.stderr)
        sys.exit(1)

    oai = OpenAI(api_key=key)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    # Staro zbirko pobrišemo, da je vsak zagon enak (brez podvajanja)
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Obstoječa zbirka '{COLLECTION_NAME}' izbrisana.")
    except Exception:
        pass

    col = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    total = 0
    for kb_file in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        print(f"Obdelavam {kb_file.name} …", end=" ", flush=True)
        text = kb_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, kb_file.name)
        if not chunks:
            print("brez vsebine, preskakujem.")
            continue

        # Dodajamo po 50 naenkrat, da ne trčimo v omejitve API-ja
        BATCH = 50
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            texts_b = [c["text"] for c in batch]
            metas_b = [c["metadata"] for c in batch]
            ids_b   = [f"{kb_file.stem}_{c['metadata']['chunk_id']}" for c in batch]
            embeds  = embed_batch(texts_b, oai)
            col.add(
                documents=texts_b,
                embeddings=embeds,
                metadatas=metas_b,
                ids=ids_b,
            )

        print(f"{len(chunks)} delcev")
        total += len(chunks)

    print(f"\nSkupaj delcev v zbirki: {total}")


if __name__ == "__main__":
    main()

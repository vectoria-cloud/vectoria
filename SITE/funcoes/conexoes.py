"""Módulo de conexões/caches (Neo4j, Chroma, embedder, warm-up de modelos).

Objetivo:
  - manter conexões vivas enquanto o site estiver rodando (evitar reconectar por request)
  - expor getters thread-safe (Neo4j driver é thread-safe; use session por execução)
"""

from __future__ import annotations

import atexit
import os
from threading import Lock
from typing import Any

# -------------------------
# Config por variáveis de ambiente
# -------------------------
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "vectoria")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "zWF$yls*J;K:DtC3")
NEO4J_MAX_POOL_SIZE = int(os.getenv("NEO4J_MAX_POOL_SIZE", "20"))

CHROMA_API_KEY         = os.getenv("CHROMA_API_KEY",        "ck-13V15SvUh23Zc7MXYoio9uoNGHgyJLVNcwJw9ZxYr2Z2")
CHROMA_TENANT          = os.getenv("CHROMA_TENANT",         "c3e00254-1f1b-49fb-8f51-c9fbad3c8d76")
CHROMA_DATABASE        = os.getenv("CHROMA_DATABASE",       "arbopedia")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME","arbopedia")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Modelos no Ollama
MODEL_NAME_PLANNER = os.getenv("MODEL_NAME_PLANNER", "PLANNER")
MODEL_NAME_CYPHER = os.getenv("MODEL_NAME_CYPHER", "CYPHER_V2")
MODEL_NAME_CYPHER_CORRECTOR = os.getenv("MODEL_NAME_CYPHER_CORRECTOR", "CYPHER_CORRECTOR")
MODEL_NAME_ANSWER = os.getenv("MODEL_NAME_ANSWER", "ANSWER")

OLLAMA_DETERMINISTIC_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
}

# -------------------------
# Neo4j driver (singleton)
# -------------------------
_neo4j_driver = None
_neo4j_lock = Lock()


def init_neo4j_driver(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    max_pool_size: int = NEO4J_MAX_POOL_SIZE,
    **kwargs: Any,
):
    """Inicializa o driver Neo4j uma única vez."""
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver is None:
            try:
                from neo4j import GraphDatabase  # type: ignore
            except Exception as e:
                raise RuntimeError("Pacote 'neo4j' não instalado. Instale: pip install neo4j") from e

            _neo4j_driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_pool_size,
                **kwargs,
            )
    return _neo4j_driver


def get_neo4j_driver():
    if _neo4j_driver is None:
        init_neo4j_driver()
    return _neo4j_driver


def close_neo4j():
    global _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver is not None:
            try:
                _neo4j_driver.close()
            finally:
                _neo4j_driver = None


atexit.register(close_neo4j)

# -------------------------
# Chroma (opcional)
# -------------------------
_chroma_client = None
_chroma_collection = None
_chroma_lock = Lock()


def _chroma_configured() -> bool:
    return bool(CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE and CHROMA_COLLECTION_NAME)


def init_chroma():
    """Inicializa cliente/collection Chroma (se configurado)."""
    global _chroma_client, _chroma_collection
    if not _chroma_configured():
        return None

    with _chroma_lock:
        if _chroma_client is None or _chroma_collection is None:
            try:
                import chromadb  # type: ignore
            except Exception as e:
                raise RuntimeError("Pacote 'chromadb' não instalado. Instale: pip install chromadb") from e

            # Ajuste aqui conforme seu setup (CloudClient/HttpClient etc.)
            try:
                _chroma_client = chromadb.CloudClient(
                    api_key=CHROMA_API_KEY,
                    tenant=CHROMA_TENANT,
                    database=CHROMA_DATABASE,
                )
            except Exception:
                _chroma_client = chromadb.HttpClient(
                    host=os.getenv("CHROMA_HOST", ""),
                    port=int(os.getenv("CHROMA_PORT", "443")),
                    ssl=True,
                    headers={"x-chroma-token": CHROMA_API_KEY},
                )

            _chroma_collection = _chroma_client.get_collection(CHROMA_COLLECTION_NAME)

    return _chroma_collection


def get_chroma_collection():
    if not _chroma_configured():
        return None
    if _chroma_collection is None:
        init_chroma()
    return _chroma_collection

# -------------------------
# Embedder (opcional)
# -------------------------
_embedder = None
_embed_lock = Lock()


def get_embedder():
    global _embedder
    with _embed_lock:
        if _embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Pacote 'sentence-transformers' não instalado. Instale: pip install sentence-transformers"
                ) from e
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

# -------------------------
# Warm-up (Ollama) opcional
# -------------------------
_warmed = False
_warm_lock = Lock()


def warm_up_ollama():
    """Dá 'ping' nos modelos do Ollama pra evitar latência na primeira pergunta."""
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        try:
            import ollama  # type: ignore
            for m in [MODEL_NAME_PLANNER, MODEL_NAME_CYPHER, MODEL_NAME_CYPHER_CORRECTOR, MODEL_NAME_ANSWER]:
                try:
                    ollama.chat(
                        model=m,
                        messages=[{"role": "user", "content": "ping"}],
                        options=OLLAMA_DETERMINISTIC_OPTIONS,
                    )
                except Exception:
                    pass
        finally:
            _warmed = True


def init_connections():
    """Chame 1x no startup do Flask."""
    init_neo4j_driver()
    if _chroma_configured():
        init_chroma()
        try:
            get_embedder()
        except Exception:
            pass
    warm_up_ollama()

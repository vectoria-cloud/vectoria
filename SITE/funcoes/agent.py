from __future__ import annotations

import os
import re
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase, basic_auth
import chromadb
import ollama


# =========================
# CONFIG
# =========================

MODEL_NAME_PLANNER = os.getenv("MODEL_NAME_PLANNER", "PLANNER")
MODEL_NAME_CYPHER  = os.getenv("MODEL_NAME_CYPHER", "CYPHER_GENERATOR")
MODEL_NAME_ANSWER  = os.getenv("MODEL_NAME_ANSWER", "ANSWER")

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "vectoria")

# ⚠️ SEM default sensível: configure via env var
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "zWF$yls*J;K:DtC3")

CHROMA_API_KEY        = os.getenv("CHROMA_API_KEY",        "ck-13V15SvUh23Zc7MXYoio9uoNGHgyJLVNcwJw9ZxYr2Z2")
CHROMA_TENANT         = os.getenv("CHROMA_TENANT",         "c3e00254-1f1b-49fb-8f51-c9fbad3c8d76")
CHROMA_DATABASE        = os.getenv("CHROMA_DATABASE", "arbopedia")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "arbopedia")

LOGS_BASE_DIR = Path(os.getenv("LOGS_BASE_DIR", "logs_sessions"))
LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

# onde este arquivo (agent.py) está
THIS_DIR = Path(__file__).resolve().parent

# se não tiver env var, usa funcoes/schema_vetorizado
SCHEMA_INDEX_DIR = Path(os.getenv("SCHEMA_INDEX_DIR", str(THIS_DIR / "schema_vetorizado"))).resolve()

SCHEMA_EMB_PATH    = SCHEMA_INDEX_DIR / "schema_emb.npy"
SCHEMA_CHUNKS_PATH = SCHEMA_INDEX_DIR / "schema_chunks.json"

# DEBUG
DEBUG_IA2 = os.getenv("DEBUG_IA2", "1").strip() in ("1", "true", "True", "YES", "yes")
DEBUG_PRINT_MAX_CHARS = int(os.getenv("DEBUG_PRINT_MAX_CHARS", "2500"))
SAVE_DEBUG_JSON = os.getenv("SAVE_DEBUG_JSON", "0").strip() in ("1", "true", "True", "YES", "yes")


# =========================
# LOG
# =========================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")

def _dbg_block(title: str, content: Any, max_chars: int = DEBUG_PRINT_MAX_CHARS) -> None:
    """
    Print de bloco (truncado) para debug. Nunca imprime segredos.
    """
    try:
        s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2, default=str)
    except Exception:
        s = str(content)
    s = s or ""
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... (truncado, total={len(s)} chars)"
    print(f"\n[DEBUG] ===== {title} =====\n{s}\n[DEBUG] ===== /{title} =====\n")


# =========================
# JSON SAFE
# =========================

def json_safe(x: Any) -> Any:
    """
    Converte objetos não-serializáveis (datas, numpy, Path, etc) em formatos JSON-safe.
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (date, datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    try:
        return str(x)
    except Exception:
        return repr(x)


# =========================
# HELPERS
# =========================

def new_session_id() -> str:
    return uuid.uuid4().hex[:12]

def prepare_text_for_model(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

def safe_json_loads(text: str) -> Any:
    """
    Faz o possível para transformar um texto em JSON (mesmo se vier com fences).
    """
    text = (text or "").strip()

    # 1) direto
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) remove ```json ... ``` ou ``` ... ```
    m = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass

    # 3) primeiro {...} bem-formado
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Não consegui interpretar como JSON. Amostra: {text[:200]}")


# =========================
# EMBEDDINGS (RAG)
# =========================

_embed_lock = threading.Lock()
_embed_model: Optional[SentenceTransformer] = None

def _load_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                log_info(f"Carregando modelo de embedding: {EMBED_MODEL_NAME}")
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def embed_normative_query(text: str) -> List[float]:
    prepared = prepare_text_for_model(text)
    model = _load_embed_model()
    vec = model.encode(prepared, normalize_embeddings=True)
    return vec.astype(float).tolist()


# =========================
# CHROMA CLOUD (RAG normativo)
# =========================

_chroma_client = None
_rag_collection = None
_chroma_lock = threading.Lock()

def _get_rag_collection():
    """
    Lazy init para evitar falhas de import/boot quando RAG não é usado.
    """
    global _chroma_client, _rag_collection
    if _rag_collection is not None:
        return _rag_collection

    with _chroma_lock:
        if _rag_collection is not None:
            return _rag_collection

        if not (CHROMA_API_KEY and CHROMA_TENANT):
            log_warn("CHROMA_API_KEY/CHROMA_TENANT não configurados. RAG ficará indisponível.")
            _rag_collection = None
            return None

        _chroma_client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
        _rag_collection = _chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        return _rag_collection

def _flatten_chroma_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        return x[0]
    if isinstance(x, list):
        return x
    return []

def normalize_chroma_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma Cloud pode exigir 1 operador no topo. Se vierem múltiplos campos no topo,
    embrulha em {"$and": [{"k": v}, ...]}.
    """
    if not where or not isinstance(where, dict):
        return None

    top_keys = list(where.keys())
    if len(top_keys) == 1 and isinstance(top_keys[0], str) and top_keys[0].startswith("$"):
        return where
    if len(top_keys) == 1:
        return where
    return {"$and": [{k: where[k]} for k in top_keys]}

def run_rag_for_question(
    question: str,
    rag_plan: Optional[Dict[str, Any]] = None,
    n_results: int = 5,
    rag_query_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Busca documentos normativos no Chroma Cloud.
    Faz fallback sem where se o where do planner não retornar nada.
    """
    collection = _get_rag_collection()
    if collection is None:
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "where_used": None,
            "where_original": None,
            "rag_fallback_used": False,
            "query_used": question,
            "error": "RAG indisponível (Chroma não configurado).",
        }

    where_original: Optional[Dict[str, Any]] = None
    if rag_plan and isinstance(rag_plan, dict):
        where_original = rag_plan.get("where")
    where_original = normalize_chroma_where(where_original)

    query_text = question
    if rag_query_hint:
        query_text = f"{question}\n\nCONTEXTO DO GRAFO (RESUMO):\n{rag_query_hint}"

    def _query(where_used: Optional[Dict[str, Any]], qt: str) -> Dict[str, Any]:
        q_emb = embed_normative_query(qt)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where_used or None,
        )
        docs_raw = results.get("documents", [[]])
        metas_raw = results.get("metadatas", [[]])
        ids_raw = results.get("ids", [[]])
        dists_raw = results.get("distances", [[]])

        return {
            "documents": docs_raw[0] if docs_raw else [],
            "metadatas": metas_raw[0] if metas_raw else [],
            "ids": ids_raw[0] if ids_raw else [],
            "distances": dists_raw[0] if dists_raw else [],
            "where_used": where_used,
            "query_used": qt,
            "error": None,
        }

    try:
        r1 = _query(where_original, query_text)
        if _flatten_chroma_list(r1.get("ids")):
            r1["where_original"] = where_original
            r1["rag_fallback_used"] = False
            return r1

        r2 = _query(None, query_text)
        r2["where_original"] = where_original
        r2["rag_fallback_used"] = True
        r2["fallback_reason"] = "0 resultados com where_original"
        return r2

    except Exception as e:
        log_error(f"Falha ao consultar Chroma Cloud: {e}")
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "where_used": where_original,
            "where_original": where_original,
            "rag_fallback_used": False,
            "query_used": query_text,
            "error": str(e),
        }


# =========================
# MINI-RAG DE SCHEMA (para IA 2)
# =========================

_schema_embeddings: Optional[np.ndarray] = None
_schema_chunks: Optional[List[Dict[str, Any]]] = None
_schema_lock = threading.Lock()

def _load_schema_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    global _schema_embeddings, _schema_chunks
    if _schema_embeddings is not None and _schema_chunks is not None:
        return _schema_embeddings, _schema_chunks

    with _schema_lock:
        if _schema_embeddings is not None and _schema_chunks is not None:
            return _schema_embeddings, _schema_chunks

        if SCHEMA_EMB_PATH.exists() and SCHEMA_CHUNKS_PATH.exists():
            log_info("Carregando índice vetorizado do schema...")
            _schema_embeddings = np.load(SCHEMA_EMB_PATH)
            _schema_chunks = json.loads(SCHEMA_CHUNKS_PATH.read_text(encoding="utf-8"))
        else:
            log_warn("Arquivos de mini-RAG de schema não encontrados; IA2 ficará mais genérica.")
            _schema_embeddings = np.zeros((0, 1024), dtype=float)
            _schema_chunks = []

        return _schema_embeddings, _schema_chunks

def mini_rag_schema(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    embs, chunks = _load_schema_index()
    if embs.shape[0] == 0:
        return []
    q_emb = np.array(embed_normative_query(query), dtype=float)
    scores = embs @ q_emb
    idxs = np.argsort(-scores)[:top_k]
    return [chunks[int(i)] for i in idxs]

def retrieve_schema_context_for_cypher(
    question: str,
    aischema: Dict[str, Any],
    top_k: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    query = f"{question}\n\nAISchema:\n{json.dumps(aischema, ensure_ascii=False)}"
    schema_chunks = mini_rag_schema(query, top_k=top_k)
    if not schema_chunks:
        return "(schema context não encontrado — Cypher será gerado de forma mais genérica)", []
    docs_str = "\n\n".join(
        f"[CHUNK {i+1}] {(c.get('text') or c.get('content') or '').strip()}"
        for i, c in enumerate(schema_chunks)
    )
    return docs_str.strip(), schema_chunks


# =========================
# NEO4J (read-only)
# =========================

_neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD),
    encrypted=False,
    max_connection_lifetime=3600,
)

def run_cypher_safe(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    params = params or {}
    log_info(f"Executando Cypher (read-only) em DB={NEO4J_DATABASE}: {cypher[:180]}...")
    with _neo4j_driver.session(
        database=NEO4J_DATABASE,
        default_access_mode="READ"
    ) as session:
        try:
            result = session.run(cypher, **params)
            rows = [r.data() for r in result]
            log_info(f"Neo4j retornou {len(rows)} linha(s).")
            return rows
        except Exception as e:
            log_error(f"Erro ao executar Cypher: {e}")
            return [{"error": str(e)}]


# =========================
# IA 1 — PLANNER
# =========================

_schema_llm_lock = threading.Lock()

def infer_needs_flags_from_schema(data: Dict[str, Any]) -> Tuple[str, bool, bool]:
    backend_mode = data.get("backend_mode") or data.get("backend") or "graph_only"
    backend_mode = backend_mode.strip().lower()

    needs_query = bool(data.get("needs_query"))
    needs_query_rag = bool(data.get("needs_query_rag"))

    # fallback
    if backend_mode == "graph_only":
        needs_query = True
        needs_query_rag = False
    elif backend_mode == "rag_only":
        needs_query = False
        needs_query_rag = True
    elif backend_mode == "hybrid_graph_rag":
        needs_query = True
        needs_query_rag = True

    return backend_mode, needs_query, needs_query_rag

def get_ai_schema_dict(question: str) -> Dict[str, Any]:
    with _schema_llm_lock:
        start = time.time()
        planner_raw = None
        planner_error = None
        try:
            resp = ollama.chat(
                model=MODEL_NAME_PLANNER,
                messages=[{"role": "user", "content": question}],
                format="json",
            )
            planner_raw = (resp.get("message", {}) or {}).get("content", "")
        except Exception as e:
            planner_error = str(e)
            planner_raw = ""

        elapsed = time.time() - start

    data: Dict[str, Any]
    try:
        data_any = safe_json_loads(planner_raw or "{}")
        data = data_any if isinstance(data_any, dict) else {"raw": data_any}
    except Exception as e:
        data = {"raw_text": planner_raw, "parse_error": str(e)}

    backend_mode, needs_query, needs_query_rag = infer_needs_flags_from_schema(data)
    data["backend_mode"] = backend_mode
    data["needs_query"] = needs_query
    data["needs_query_rag"] = needs_query_rag
    data.setdefault("rag_plan", None)
    data["_generation_time"] = elapsed
    data["_planner_raw"] = planner_raw[:8000] if isinstance(planner_raw, str) else str(planner_raw)[:8000]
    data["_planner_error"] = planner_error
    return data


# =========================
# IA 2 — CYPHER_GENERATOR
# =========================

def sanitize_cypher(q: str) -> str:
    if not q:
        return q

    q = q.strip()
    q = re.sub(r"^\s*```(?:cypher)?\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*```\s*$", "", q)
    q = re.sub(r"^\s*cypher\s*:\s*", "", q, flags=re.IGNORECASE)
    q = q.strip().strip("`").strip()
    q = q.rstrip(";").strip()
    q = q.strip().strip('"').strip("'").strip()
    return q

def is_cypher_suspicious(q: str) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not q:
        return True, ["cypher_empty"]
    uq = q.upper()

    if "MATCH P =" in uq or "MATCH P=" in uq:
        issues.append("uses_path_variable_p")
    if "SUM(P." in uq or "AVG(P." in uq or "COUNT(P." in uq:
        issues.append("aggregates_over_path_p")
    if "RETURN DISTINCT" in uq and ("SUM(" in uq or "AVG(" in uq or "COUNT(" in uq):
        issues.append("distinct_with_aggregation")
    if "ATIVIDADERESUMO" in uq:
        issues.append("uses_AtividadeResumo_label")
    if "CREATE " in uq or "DELETE " in uq or "MERGE " in uq or "SET " in uq:
        issues.append("write_query_detected")

    return (len(issues) > 0), issues

def cypher_is_read_only(q: str) -> bool:
    if not q:
        return True
    bad = [
        r"\bCREATE\b", r"\bMERGE\b", r"\bDELETE\b", r"\bDETACH\b",
        r"\bSET\b", r"\bDROP\b", r"\bREMOVE\b", r"\bCALL\s+db\.",
        r"\bLOAD\s+CSV\b", r"\bAPOC\."
    ]
    return not any(re.search(pat, q, flags=re.IGNORECASE) for pat in bad)

def try_repair_common_cypher_issues(q: str) -> str:
    if not q:
        return q
    qq = q.strip()

    def _wrap_date(m):
        op = m.group(1)
        dt = m.group(2)
        return f"{op} date('{dt}')"

    qq = re.sub(
        r"(\b>=\s*|\b<=\s*|\b=\s*)(\d{4}-\d{2}-\d{2})\b",
        _wrap_date,
        qq
    )

    qq = re.sub(
        r"\bBETWEEN\s+(\d{4}-\d{2}-\d{2})\s+AND\s+(\d{4}-\d{2}-\d{2})\b",
        lambda m: f"BETWEEN date('{m.group(1)}') AND date('{m.group(2)}')",
        qq,
        flags=re.IGNORECASE
    )
    return qq

def build_cypher_prompt(
    question: str,
    aischema: Dict[str, Any],
    schema_context: str,
) -> str:
    templates = """
TEMPLATES CORRETOS (prefira estes padrões):

(1) Backbone Município -> Dia
MATCH (m:Municipio)-[:TEM_DADO_NO_DIA]->(d:Dia)

(2) Casos por agravo
MATCH (m)-[:TEM_CASOS]->(c:Casos)
MATCH (d)-[:TEM_CASOS]->(c)
MATCH (c)-[:E_DO_AGRAVO]->(a:Agravo)

(3) Agregação mensal segura
RETURN date.truncate('month', d.date) AS periodo, m.nome AS municipio, sum(c.qtd) AS total
ORDER BY periodo ASC, municipio

(4) Meteorologia
MATCH (d)-[:TEM_METEOROLOGIA]->(me:Meteo)

(5) Atividades
MATCH (m)-[:EXECUTOU_ATIVIDADE]->(ae:AtividadeExec)
MATCH (ae)-[:NO_DIA]->(d:Dia)

(6) Notificações
MATCH (d)-[:TEM_NOTIFICACAO]->(n:Notificacao)
MATCH (n)-[:E_DO_AGRAVO]->(a:Agravo)
""".strip()

    hard_rules = """
REGRAS OBRIGATÓRIAS:
- Retorne APENAS a Cypher final (texto puro). NÃO use ``` e NÃO use `...`.
- NÃO use MATCH p = ... e nunca agregue (sum/avg/count) sobre Path.
- Evite RETURN DISTINCT com SUM/AVG/COUNT (use WITH DISTINCT antes, se precisar).
- ORDER BY só pode usar variáveis/aliases presentes no MESMO WITH/RETURN.
- Não invente labels/relacionamentos fora do mini-RAG.
- Query READ-ONLY: sem CREATE/MERGE/SET/DELETE.
""".strip()

    return f"""
Você é a IA 2 (CYPHER_GENERATOR), especialista em gerar consultas Cypher para Neo4j.

{hard_rules}

Pergunta:
{question}

AISchema (IA 1):
{json.dumps(aischema, ensure_ascii=False, indent=2)}

Mini-RAG (contexto do grafo):
{schema_context}

{templates}

Tarefa:
- Gere a Cypher mais simples possível que responda ao AISchema.
- Respeite filtros (município, datas, agravos) e granularidade.
- Retorne somente a Cypher final.
""".strip()

def get_cypher_from_schema(
    question: str,
    aischema: Dict[str, Any],
) -> Tuple[str, List[str], Dict[str, Any]]:
    schema_context, schema_chunks = retrieve_schema_context_for_cypher(question, aischema, top_k=6)
    prompt = build_cypher_prompt(question, aischema, schema_context=schema_context)

    raw = ""
    ia2_error = None
    try:
        resp = ollama.chat(
            model=MODEL_NAME_CYPHER,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.get("message", {}) or {}).get("content", "") or ""
    except Exception as e:
        ia2_error = str(e)
        raw = ""

    # extrai de fences se vier
    m = re.search(r"```cypher(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE) or re.search(r"```(.*?)```", raw, flags=re.DOTALL)
    cypher = (m.group(1).strip() if m else raw.strip())
    cypher_sanitized = sanitize_cypher(cypher)
    cypher_repaired = try_repair_common_cypher_issues(cypher_sanitized)

    warnings: List[str] = []
    suspicious, issues = is_cypher_suspicious(cypher_sanitized)
    if suspicious:
        warnings.extend(issues)

    read_only = cypher_is_read_only(cypher_repaired)

    # DEBUG PRINTS (IA2)
    if DEBUG_IA2:
        _dbg_block("IA2.schema_context", schema_context)
        _dbg_block("IA2.schema_chunks_count", {"count": len(schema_chunks)})
        if schema_chunks:
            # imprime só um resumo dos chunks para não explodir
            chunks_preview = []
            for i, c in enumerate(schema_chunks[:6], start=1):
                txt = (c.get("text") or c.get("content") or "")
                chunks_preview.append({
                    "i": i,
                    "keys": list(c.keys()),
                    "preview": (txt[:300] + ("..." if len(txt) > 300 else "")),
                })
            _dbg_block("IA2.schema_chunks_preview", chunks_preview)

        _dbg_block("IA2.prompt", prompt)
        _dbg_block("IA2.raw_model_output", raw)
        _dbg_block("IA2.cypher_sanitized", cypher_sanitized)
        _dbg_block("IA2.cypher_repaired", cypher_repaired)
        _dbg_block("IA2.warnings", warnings)
        _dbg_block("IA2.read_only", {"read_only": read_only})
        if ia2_error:
            _dbg_block("IA2.error", ia2_error)

    debug = {
        "schema_context": schema_context,
        "schema_chunks": schema_chunks,
        "prompt": prompt,
        "raw_model_output": raw[:8000],
        "ia2_error": ia2_error,
        "cypher_sanitized": cypher_sanitized,
        "cypher_repaired": cypher_repaired,
        "warnings": warnings,
        "read_only": read_only,
    }
    return cypher_repaired, warnings, debug


# =========================
# GRAFO -> RESUMO (p/ híbrido e p/ tabela na resposta)
# =========================

def build_graph_preview_table(graph_rows: List[Dict[str, Any]], max_rows: int = 50) -> str:
    if not graph_rows:
        return "(sem dados do grafo ou consulta não executada)"
    header = list(graph_rows[0].keys())
    lines = [" | ".join(header)]
    for r in graph_rows[:max_rows]:
        row = [str(r.get(h, "")) for h in header]
        lines.append(" | ".join(row))
    return "\n".join(lines)

def build_graph_summary_for_rag(
    question: str,
    aischema: Dict[str, Any],
    cypher: Optional[str],
    graph_rows: List[Dict[str, Any]],
    neo4j_error: Optional[str],
    max_chars: int = 2000,
) -> str:
    parts = []
    parts.append("=== CONTEXTO DO GRAFO (RESUMO) ===")
    parts.append(f"Pergunta: {question}")

    filtros = (aischema or {}).get("filters", {}) or {}
    targets = (aischema or {}).get("targets", {}) or {}
    municipios = filtros.get("municipios") or []
    datef = filtros.get("date") or {}
    agravos = (targets.get("agravos") or []) or (filtros.get("agravos") or [])

    if municipios:
        parts.append(f"Município(s): {municipios}")
    if agravos:
        parts.append(f"Agravos/Doenças: {agravos}")
    if datef:
        parts.append(f"Filtro de data: {datef}")

    if cypher:
        parts.append("Cypher executada (higienizada):")
        parts.append(cypher)

    if neo4j_error:
        parts.append("STATUS: ERRO ao executar no Neo4j.")
        parts.append(f"Erro: {neo4j_error}")
    else:
        parts.append("STATUS: OK (sem erro Neo4j reportado).")

    if graph_rows:
        parts.append("Amostra (primeiras linhas):")
        parts.append(build_graph_preview_table(graph_rows, max_rows=10))
    else:
        parts.append("Amostra: (sem linhas retornadas)")

    txt = "\n".join(parts)
    return txt[:max_chars]

def merge_rag_query_with_graph_context(question: str, graph_summary: str) -> str:
    if not graph_summary:
        return question
    return (
        f"{question}\n\n{graph_summary}\n\n"
        "Tarefa do retrieval: recuperar trechos normativos diretamente relacionados "
        "aos achados acima (ou à ausência deles), priorizando recomendações operacionais."
    )


# =========================
# IA 3 — ANSWER (resposta final)
# =========================

def build_final_answer(
    question: str,
    aischema: Dict[str, Any],
    backend_mode: str,
    graph_rows: List[Dict[str, Any]],
    rag_context: Dict[str, Any],
    neo4j_error: Optional[str] = None,
) -> str:
    graph_preview = build_graph_preview_table(graph_rows, max_rows=80)

    rag_docs = rag_context.get("documents") or []
    rag_metas = rag_context.get("metadatas") or []
    rag_ids = rag_context.get("ids") or []

    rag_snippets = []
    for i, (doc, meta, rid) in enumerate(zip(rag_docs, rag_metas, rag_ids), start=1):
        src = meta.get("source") if isinstance(meta, dict) else meta
        rag_snippets.append(f"[DOC {i} | id={rid} | fonte={src}]\n{doc}")

    rag_block = "\n\n".join(rag_snippets) if rag_snippets else "(nenhum trecho normativo recuperado)"

    prompt = (
        "Você é a IA 3 (ANSWER), responsável por produzir a resposta final ao usuário, "
        "integrando resultados do grafo de conhecimento (se houver) com trechos normativos recuperados (se houver).\n\n"
        f"Pergunta original do usuário:\n{question}\n\n"
        "Plano produzido pela IA 1 (AISchema):\n"
        f"{json.dumps(aischema, ensure_ascii=False, indent=2)}\n\n"
        f"Backend selecionado: {backend_mode}\n\n"
        f"Neo4j_error (se houver): {neo4j_error}\n\n"
        "Prévia dos resultados do grafo (primeiras linhas):\n"
        f"{graph_preview}\n\n"
        "Trechos normativos recuperados (RAG):\n"
        f"{rag_block}\n\n"
        "Tarefa:\n"
        "- Responda de forma clara, técnica e rastreável;\n"
        "- Quando houver série temporal, apresente em tabela (período, município, métrica) e diga a unidade/definição se estiver no grafo;\n"
        "- Em cenário híbrido, conecte explicitamente o diagnóstico quantitativo (grafo) com recomendações normativas (RAG);\n"
        "- Se o grafo falhar (neo4j_error) ou não retornar linhas, deixe isso explícito e NÃO invente números;\n"
        "- Se o RAG não retornar trechos, deixe explícito e NÃO cite diretrizes inexistentes;\n"
        "- Quando usar evidência normativa, cite pelo marcador [DOC X] e id;\n"
        "- Não invente entidades, relações, datas ou métricas."
    )

    try:
        resp = ollama.chat(
            model=MODEL_NAME_ANSWER,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        return f"[ERRO IA3] {e}"


# =========================
# PIPELINE (IA1..IA3) -> JSON completo
# =========================

def ask_arbo_agent_ate_ia3(
    question: str,
    session_id: Optional[str] = None,
    tag: Optional[str] = None,
    k_retrieval: int = 5,
) -> Dict[str, Any]:
    """
    Executa IA1..IA3 e devolve um dict com artefatos + final_answer.
    """
    if session_id is None:
        session_id = new_session_id()

    timers: Dict[str, float] = {}
    t0 = time.time()

    # IA1
    t_planner = time.time()
    aischema = get_ai_schema_dict(question)
    timers["planner_s"] = time.time() - t_planner

    backend_mode = (aischema.get("backend_mode") or "graph_only").strip().lower()

    needs_query_graph = bool(
        aischema.get("needs_query_graph")
        or (backend_mode in ("graph_only", "hybrid_graph_rag") and aischema.get("needs_query"))
    )
    needs_query_rag = bool(
        aischema.get("needs_query_rag")
        or (backend_mode in ("rag_only", "hybrid_graph_rag"))
    )

    if backend_mode == "hybrid_graph_rag":
        needs_query_graph = True
        needs_query_rag = True

    rag_plan = aischema.get("rag_plan") or {}

    cypher: Optional[str] = None
    cypher_warnings: List[str] = []
    cypher_debug: Dict[str, Any] = {}
    graph_result: List[Dict[str, Any]] = []
    neo4j_error: Optional[str] = None

    # IA2 + Neo4j
    if backend_mode in ("graph_only", "hybrid_graph_rag") and needs_query_graph:
        t_cypher = time.time()
        cypher, cypher_warnings, cypher_debug = get_cypher_from_schema(question, aischema)
        timers["cypher_s"] = time.time() - t_cypher

        if not cypher:
            neo4j_error = "IA2 retornou cypher vazia."
            graph_result = []
        elif not cypher_is_read_only(cypher):
            neo4j_error = "Cypher bloqueada por guardrail (não-read-only)."
            graph_result = []
        else:
            t_neo = time.time()
            graph_result = run_cypher_safe(cypher)
            timers["neo4j_s"] = time.time() - t_neo
            if graph_result and isinstance(graph_result[0], dict) and "error" in graph_result[0]:
                neo4j_error = graph_result[0].get("error")

        if DEBUG_IA2:
            _dbg_block("IA2.final_cypher_used_for_neo4j", cypher or "")
            _dbg_block("IA2.neo4j_error", neo4j_error or None)
            _dbg_block("IA2.neo4j_result_preview", graph_result[:5])

    # RAG normativo
    rag_result: Dict[str, Any] = {}
    if backend_mode in ("rag_only", "hybrid_graph_rag") and needs_query_rag:
        t_rag = time.time()
        if backend_mode == "hybrid_graph_rag":
            graph_summary = build_graph_summary_for_rag(
                question=question,
                aischema=aischema,
                cypher=cypher,
                graph_rows=graph_result,
                neo4j_error=neo4j_error,
                max_chars=2000,
            )
            rag_question = merge_rag_query_with_graph_context(question, graph_summary)
        else:
            rag_question = question

        rag_result = run_rag_for_question(
            rag_question,
            rag_plan=rag_plan,
            n_results=k_retrieval,
        )
        timers["rag_s"] = time.time() - t_rag

    # IA3
    t_ans = time.time()
    final_answer = build_final_answer(
        question=question,
        aischema=aischema,
        backend_mode=backend_mode,
        graph_rows=graph_result,
        rag_context=rag_result,
        neo4j_error=neo4j_error,
    )
    timers["answer_s"] = time.time() - t_ans
    timers["total_s"] = time.time() - t0

    pipeline_output: Dict[str, Any] = {
        "session_id": session_id,
        "tag": tag,
        "question": question,
        "aischema": aischema,
        "backend_mode": backend_mode,
        "needs_query_graph": needs_query_graph,
        "needs_query_rag": needs_query_rag,

        # IA2
        "cypher": cypher,
        "cypher_warnings": cypher_warnings,
        "cypher_debug": {
            "schema_context": cypher_debug.get("schema_context"),
            "schema_chunks": cypher_debug.get("schema_chunks"),
            "prompt": cypher_debug.get("prompt"),
            "raw_model_output": cypher_debug.get("raw_model_output"),
            "ia2_error": cypher_debug.get("ia2_error"),
            "cypher_sanitized": cypher_debug.get("cypher_sanitized"),
            "cypher_repaired": cypher_debug.get("cypher_repaired"),
            "warnings": cypher_debug.get("warnings"),
            "read_only": cypher_debug.get("read_only"),
        },

        # Execução
        "graph_result": graph_result,
        "neo4j_error": neo4j_error,

        # RAG
        "rag_result": rag_result,

        # IA3
        "final_answer": final_answer,

        # timings
        "timers": timers,
    }

    pipeline_output_safe = json_safe(pipeline_output)

    if SAVE_DEBUG_JSON:
        try:
            out_path = LOGS_BASE_DIR / f"{session_id}.json"
            out_path.write_text(json.dumps(pipeline_output_safe, ensure_ascii=False, indent=2), encoding="utf-8")
            log_info(f"DEBUG JSON salvo em: {out_path}")
        except Exception as e:
            log_warn(f"Falha ao salvar debug JSON: {e}")

    return pipeline_output_safe


# =========================
# API para o seu Flask
# =========================

def generate_response(
    message: str,
    session_id: Optional[str] = None,
    user_key: Optional[str] = None,
    k_retrieval: int = 5,
) -> str:
    """
    Mantém compatibilidade: retorna SOMENTE a resposta final (IA3).
    """
    if not message:
        return ""

    tag = user_key or None

    out = ask_arbo_agent_ate_ia3(
        question=message,
        session_id=session_id,
        tag=tag,
        k_retrieval=k_retrieval,
    )
    return out.get("final_answer", "") or ""

def generate_response_json(
    message: str,
    session_id: Optional[str] = None,
    user_key: Optional[str] = None,
    k_retrieval: int = 5,
) -> str:
    """
    NOVO: retorna JSON (string) com tudo do fluxo IA1..IA3.
    Perfeito pra debugar no Flask/cliente.
    """
    if not message:
        return json.dumps({"error": "message vazio"}, ensure_ascii=False)

    tag = user_key or None
    out = ask_arbo_agent_ate_ia3(
        question=message,
        session_id=session_id,
        tag=tag,
        k_retrieval=k_retrieval,
    )
    return json.dumps(out, ensure_ascii=False, indent=2)


# Opcional: finalize recursos se você quiser encerrar o processo de forma limpa
def close_drivers() -> None:
    try:
        _neo4j_driver.close()
    except Exception:
        pass


if __name__ == "__main__":
    # Exemplo rápido:
    q = "Qual foi o total de casos de dengue em São José do Rio Preto no ano de 2022?"
    print(generate_response_json(q))

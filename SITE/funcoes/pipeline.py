# funcoes/pipeline.py
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

from .connections import neo4j_session, rag_collection
from .trace_logger import TraceLogger


MODEL_NAME_PLANNER = os.getenv("MODEL_NAME_PLANNER", "PLANNER")
MODEL_NAME_CYPHER  = os.getenv("MODEL_NAME_CYPHER", "CYPHER_GENERATOR")
MODEL_NAME_ANSWER  = os.getenv("MODEL_NAME_ANSWER", "ANSWER")


def safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    raise ValueError(f"Não consegui interpretar o texto como JSON. Amostra: {text[:200]}")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def prepare_text_for_model(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


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
    if "CASOS_EM*" in uq or "CASOS_NO_DIA" in uq:
        issues.append("uses_old_rel_names")
    if re.search(r"\b(CREATE|DELETE|MERGE|SET)\b", uq):
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
    return not any(re.search(p, q, flags=re.IGNORECASE) for p in bad)


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



EMBED_MODEL_NAME = "BAAI/bge-m3"
_embed_lock = threading.Lock()
_embed_model: Optional[SentenceTransformer] = None

def _load_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def embed_normative_query(text: str) -> List[float]:
    prepared = prepare_text_for_model(text)
    model = _load_embed_model()
    vec = model.encode(prepared, normalize_embeddings=True)
    return vec.astype(float).tolist()


SCHEMA_INDEX_DIR   = Path("schema_vetorizado")
SCHEMA_EMB_PATH    = SCHEMA_INDEX_DIR / "schema_emb.npy"
SCHEMA_CHUNKS_PATH = SCHEMA_INDEX_DIR / "schema_chunks.json"

_schema_embeddings: Optional[np.ndarray] = None
_schema_chunks: Optional[List[Dict[str, Any]]] = None

def _load_schema_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    global _schema_embeddings, _schema_chunks
    if _schema_embeddings is None or _schema_chunks is None:
        if SCHEMA_EMB_PATH.exists() and SCHEMA_CHUNKS_PATH.exists():
            _schema_embeddings = np.load(SCHEMA_EMB_PATH)
            with SCHEMA_CHUNKS_PATH.open("r", encoding="utf-8") as f:
                _schema_chunks = json.load(f)
        else:
            _schema_embeddings = np.zeros((0, 1024), dtype=float)
            _schema_chunks = []
    return _schema_embeddings, _schema_chunks

def mini_rag_schema(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    embs, chunks = _load_schema_index()
    if embs.shape[0] == 0:
        return []
    q_emb = np.array(embed_normative_query(query), dtype=float)
    scores = embs @ q_emb
    idxs = np.argsort(-scores)[:top_k]
    return [chunks[int(i)] for i in idxs]


def run_cypher_safe(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    params = params or {}
    with neo4j_session(access_mode="READ") as session:
        try:
            result = session.run(cypher, **params)
            return [r.data() for r in result]
        except Exception as e:
            return [{"error": str(e)}]


_schema_llm_lock = threading.Lock()

def infer_needs_flags_from_schema(data: Dict[str, Any]) -> Tuple[str, bool, bool]:
    backend_mode = (data.get("backend_mode") or data.get("backend") or "graph_only").strip().lower()

    needs_query     = bool(data.get("needs_query"))
    needs_query_rag = bool(data.get("needs_query_rag"))

    if backend_mode == "graph_only":
        needs_query, needs_query_rag = True, False
    elif backend_mode == "rag_only":
        needs_query, needs_query_rag = False, True
    elif backend_mode == "hybrid_graph_rag":
        needs_query, needs_query_rag = True, True

    return backend_mode, needs_query, needs_query_rag

def get_ai_schema_dict(question: str) -> Dict[str, Any]:
    with _schema_llm_lock:
        start = time.time()
        resp = ollama.chat(
            model=MODEL_NAME_PLANNER,
            messages=[{"role": "user", "content": question}],
            format="json",
        )
        elapsed = time.time() - start

    data = safe_json_loads(resp.get("message", {}).get("content", "{}"))
    backend_mode, needs_query, needs_query_rag = infer_needs_flags_from_schema(data)
    data["backend_mode"] = backend_mode
    data["needs_query"] = needs_query
    data["needs_query_rag"] = needs_query_rag
    if "rag_plan" not in data:
        data["rag_plan"] = None
    data["_generation_time"] = elapsed
    return data


def retrieve_schema_context_for_cypher(question: str, aischema: Dict[str, Any], top_k: int = 6) -> str:
    query = f"{question}\n\nAISchema:\n{json.dumps(aischema, ensure_ascii=False)}"
    schema_chunks = mini_rag_schema(query, top_k=top_k)

    if not schema_chunks:
        return "(schema context não encontrado — Cypher será gerado de forma mais genérica)"

    docs_str = "\n\n".join(
        f"[CHUNK {i+1}] {(c.get('text') or c.get('content') or '').strip()}"
        for i, c in enumerate(schema_chunks)
    )
    return docs_str.strip()

def build_cypher_prompt(question: str, aischema: Dict[str, Any]) -> str:
    schema_context = retrieve_schema_context_for_cypher(question, aischema, top_k=6)

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

TEMPLATES CORRETOS (prefira estes padrões):
{templates}

Tarefa:
- Gere a Cypher mais simples possível que responda ao AISchema.
- Respeite filtros (município, datas, agravos) e granularidade.
- Retorne somente a Cypher final.
""".strip()

def get_cypher_from_schema(question: str, aischema: Dict[str, Any]) -> Tuple[str, List[str]]:
    prompt = build_cypher_prompt(question, aischema)

    resp = ollama.chat(
        model=MODEL_NAME_CYPHER,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.get("message", {}).get("content", "") or ""

    m = re.search(r"```cypher(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", raw, flags=re.DOTALL)
    cypher = (m.group(1).strip() if m else raw.strip())

    cypher = sanitize_cypher(cypher)

    warnings: List[str] = []
    suspicious, issues = is_cypher_suspicious(cypher)
    if suspicious:
        warnings.extend(issues)

    return cypher, warnings



def _flatten_chroma_list(x: Any) -> List[Any]:
    """
    Chroma às vezes devolve listas aninhadas (ex.: [[id1, id2, ...]]).
    """
    if x is None:
        return []
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        return x[0]
    if isinstance(x, list):
        return x
    return []


def normalize_chroma_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normaliza o filtro `where` para o formato aceito pelo Chroma Cloud.

    Observação: em alguns deployments do Chroma, o `where` precisa ter 1 operador no topo.
    Se vierem múltiplos campos no topo, embrulhamos em:
        {"$and": [{"k1": v1}, {"k2": v2}, ...]}
    """
    if not where or not isinstance(where, dict):
        return None

    top_keys = list(where.keys())

    # Se já é um operador ($and / $or / etc), mantém
    if len(top_keys) == 1 and isinstance(top_keys[0], str) and top_keys[0].startswith("$"):
        return where

    # Se tem exatamente 1 campo no topo, mantém
    if len(top_keys) == 1:
        return where

    # Se tem múltiplos campos no topo, embrulha em $and
    clauses = [{k: where[k]} for k in top_keys]
    return {"$and": clauses}


def run_rag_for_question(
    question: str,
    rag_plan: Optional[Dict[str, Any]] = None,
    n_results: int = 5,
) -> Dict[str, Any]:
    """
    Busca documentos normativos no Chroma Cloud.

    Melhorias (vs. versão simples):
      - Normaliza `rag_plan.where` para formato aceito pelo Chroma (wrap em $and quando necessário)
      - Se a busca com `where` retornar 0 ids, aplica fallback sem filtro
      - Retorna metadados de execução (where_used, query_used, rag_fallback_used, etc.)
    """
    where_raw: Optional[Dict[str, Any]] = None
    if rag_plan and isinstance(rag_plan, dict):
        where_raw = rag_plan.get("where")

    where_original = normalize_chroma_where(where_raw)

    def _query_chroma(where_used: Optional[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        q_emb = embed_normative_query(query_text)
        results = rag_collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where_used or None,
        )

        docs  = _flatten_chroma_list(results.get("documents"))
        metas = _flatten_chroma_list(results.get("metadatas"))
        ids   = _flatten_chroma_list(results.get("ids"))
        dists = _flatten_chroma_list(results.get("distances"))

        return {
            "documents": docs,
            "metadatas": metas,
            "ids": ids,
            "distances": dists,
            "where_used": where_used,
            "query_used": query_text,
            "error": None,
        }

    query_text = question

    try:
        # 1) tentativa: com filtro do planner (normalizado)
        r1 = _query_chroma(where_original, query_text)
        ids1 = _flatten_chroma_list(r1.get("ids"))
        if ids1:
            r1["rag_fallback_used"] = False
            r1["where_original"] = where_original
            r1["where_original_raw"] = where_raw
            return r1

        # 2) fallback: sem filtro (relaxa metadados)
        r2 = _query_chroma(None, query_text)
        r2["rag_fallback_used"] = True
        r2["where_original"] = where_original
        r2["where_original_raw"] = where_raw
        r2["fallback_reason"] = "0 resultados com where_original (provável mismatch de metadata/valores)"
        return r2

    except Exception as e:
        return {
            "documents": [],
            "metadatas": [],
            "ids": [],
            "distances": [],
            "where_used": where_original,
            "where_original": where_original,
            "where_original_raw": where_raw,
            "rag_fallback_used": False,
            "query_used": query_text,
            "error": str(e),
        }


def build_graph_preview_table(graph_rows: List[Dict[str, Any]], max_rows: int = 50) -> str:
    if not graph_rows:
        return "(sem dados do grafo ou consulta não executada)"
    header = list(graph_rows[0].keys())
    lines = [" | ".join(header)]
    for r in graph_rows[:max_rows]:
        lines.append(" | ".join(str(r.get(h, "")) for h in header))
    return "\n".join(lines)

def build_graph_summary_for_rag(
    question: str,
    aischema: Dict[str, Any],
    cypher: Optional[str],
    graph_rows: List[Dict[str, Any]],
    neo4j_error: Optional[str],
    max_chars: int = 2000
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

    text = "\n".join(parts)
    return text[:max_chars]

def merge_rag_query_with_graph_context(question: str, graph_summary: str) -> str:
    if not graph_summary:
        return question
    return (
        f"{question}\n\n{graph_summary}\n\n"
        "Tarefa do retrieval: recuperar trechos normativos diretamente relacionados "
        "aos achados acima (ou à ausência deles), priorizando recomendações operacionais."
    )

def build_final_answer(
    question: str,
    aischema: Dict[str, Any],
    backend_mode: str,
    graph_rows: List[Dict[str, Any]],
    rag_context: Dict[str, Any],
    neo4j_error: Optional[str] = None,
) -> str:
    graph_preview = build_graph_preview_table(graph_rows, max_rows=80)

    rag_docs  = rag_context.get("documents") or []
    rag_metas = rag_context.get("metadatas") or []
    rag_ids   = rag_context.get("ids") or []

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
        "- Em cenário híbrido, conecte explicitamente o diagnóstico quantitativo (grafo) com recomendações normativas (RAG);\n"
        "- Se o grafo falhar (neo4j_error) ou não retornar linhas, deixe isso explícito e NÃO invente números;\n"
        "- Se o RAG não retornar trechos, deixe explícito e NÃO cite diretrizes inexistentes;\n"
        "- Quando usar evidência normativa, cite pelo marcador [DOC X] e id;\n"
        "- Não invente entidades, relações, datas ou métricas."
    )

    resp = ollama.chat(
        model=MODEL_NAME_ANSWER,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.get("message", {}).get("content", "").strip()



def ask_arbo_agent(
    question: str,
    *,
    session_id: Optional[str] = None,
    k_retrieval: int = 5,
    trace: Optional[TraceLogger] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Linha principal (IA1→IA3 + grafo/RAG):

      1) IA 1 (PLANNER) -> AISchema
      2) IA 2 (CYPHER_GENERATOR) -> Cypher (se grafo/híbrido)
      3) Neo4j (read-only) -> graph_result (se grafo/híbrido)
      4) RAG (Chroma) -> rag_result (se rag/híbrido)
      5) IA 3 (ANSWER) -> final_answer
    """
    if session_id is None:
        session_id = new_session_id()

    timers: Dict[str, float] = {}
    t0 = time.time()


    t = time.time()
    aischema = get_ai_schema_dict(question)
    timers["planner_s"] = time.time() - t
    if trace and run_id:
        trace.log_step(run_id, name="planner", model=MODEL_NAME_PLANNER, input={"question": question}, output=aischema, meta={"elapsed_s": timers["planner_s"]})

    backend_mode = aischema.get("backend_mode") or "graph_only"
    needs_query_graph = bool(aischema.get("needs_query_graph") or (backend_mode in ("graph_only", "hybrid_graph_rag") and aischema.get("needs_query")))
    needs_query_rag   = bool(aischema.get("needs_query_rag") or (backend_mode in ("rag_only", "hybrid_graph_rag")))

    rag_plan = aischema.get("rag_plan") or {}

    cypher: Optional[str] = None
    cypher_warnings: List[str] = []
    graph_result: List[Dict[str, Any]] = []
    neo4j_error: Optional[str] = None

    if backend_mode in ("graph_only", "hybrid_graph_rag") and needs_query_graph:
        t = time.time()
        cypher, cypher_warnings = get_cypher_from_schema(question, aischema)
        timers["cypher_s"] = time.time() - t
        cypher = try_repair_common_cypher_issues(sanitize_cypher(cypher))

        if trace and run_id:
            trace.log_step(
                run_id,
                name="cypher_generator",
                model=MODEL_NAME_CYPHER,
                input={"question": question, "aischema": aischema},
                output={"cypher": cypher, "warnings": cypher_warnings},
                meta={"elapsed_s": timers["cypher_s"]},
            )

        if not cypher_is_read_only(cypher):
            neo4j_error = "Cypher bloqueada por guardrail (não-read-only)."
            graph_result = []
        else:
            t = time.time()
            graph_result = run_cypher_safe(cypher)
            timers["neo4j_s"] = time.time() - t

            if graph_result and isinstance(graph_result[0], dict) and "error" in graph_result[0]:
                neo4j_error = str(graph_result[0].get("error"))

        if trace and run_id:
            trace.log_step(
                run_id,
                name="neo4j",
                input={"cypher": cypher},
                output={"rows": graph_result[:200], "rows_count": len(graph_result), "neo4j_error": neo4j_error},
                meta={"elapsed_s": timers.get("neo4j_s")},
            )


    rag_result: Dict[str, Any] = {}
    if backend_mode in ("rag_only", "hybrid_graph_rag") and needs_query_rag:
        t = time.time()
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

        rag_result = run_rag_for_question(rag_question, rag_plan=rag_plan, n_results=k_retrieval)
        timers["rag_s"] = time.time() - t

        if trace and run_id:
            trace.log_step(
                run_id,
                name="rag",
                input={"rag_question": rag_question, "rag_plan": rag_plan, "k": k_retrieval},
                output={
                    "ids": rag_result.get("ids"),
                    "metadatas": rag_result.get("metadatas"),
                    "documents_preview": (rag_result.get("documents") or [])[:2],
                    "where_used": rag_result.get("where_used"),
                    "where_original": rag_result.get("where_original"),
                    "rag_fallback_used": rag_result.get("rag_fallback_used"),
                    "fallback_reason": rag_result.get("fallback_reason"),
                    "query_used": rag_result.get("query_used"),
                    "error": rag_result.get("error"),
                },
                meta={"elapsed_s": timers["rag_s"]},
            )

    t = time.time()
    final_answer = build_final_answer(
        question=question,
        aischema=aischema,
        backend_mode=backend_mode,
        graph_rows=graph_result,
        rag_context=rag_result,
        neo4j_error=neo4j_error,
    )
    timers["answer_s"] = time.time() - t
    timers["total_s"] = time.time() - t0

    if trace and run_id:
        trace.log_step(run_id, name="answer", model=MODEL_NAME_ANSWER, input={"backend_mode": backend_mode}, output={"final_answer": final_answer}, meta={"elapsed_s": timers["answer_s"]})

    return {
        "session_id": session_id,
        "question": question,
        "aischema": aischema,
        "backend_mode": backend_mode,
        "needs_query_graph": needs_query_graph,
        "needs_query_rag": needs_query_rag,
        "cypher": cypher,
        "cypher_warnings": cypher_warnings,
        "graph_result": graph_result,
        "rag_result": rag_result,
        "neo4j_error": neo4j_error,
        "final_answer": final_answer,
        "timers": timers,
    }

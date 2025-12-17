"""Fluxo do Agent (execução), separado das conexões e do logger.

- IA1 (PLANNER): sem contexto
- IA2 (CYPHER): sem contexto
- Neo4j: executa cypher
- (Opcional) RAG Chroma: executa retrieval
- IA3 (ANSWER): com contexto SOMENTE do último turno, para gerar a resposta final
- Logger: salva pergunta, respostas de cada IA, cypher, resultado neo4j e rag, resposta final
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .conexoes import (
    get_neo4j_driver,
    get_chroma_collection,
    get_embedder,
    NEO4J_DATABASE,
    MODEL_NAME_PLANNER,
    MODEL_NAME_CYPHER,
    MODEL_NAME_CYPHER_CORRECTOR,
    MODEL_NAME_ANSWER,
    OLLAMA_DETERMINISTIC_OPTIONS,
)
from .logger_json import append_entry, get_last_turn_context, now_iso


# -------------------------
# JSON safe helpers
# -------------------------
def _to_jsonable(value: Any) -> Any:
    """Converte tipos do Neo4j (Date/DateTime/Node etc.) e outros objetos em algo serializável."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    # Neo4j temporal costuma ter iso_format()
    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        try:
            return iso_format()
        except Exception:
            pass

    # datetime/date/time python tem isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    # Neo4j Node/Relationship/Path etc: tenta dict() ou vars()
    try:
        return _to_jsonable(dict(value))  # type: ignore[arg-type]
    except Exception:
        pass

    return str(value)


def _jd(obj: Any) -> str:
    """json.dumps seguro para usar dentro de prompts."""
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)


# -------------------------
# Helpers
# -------------------------
def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


def _ollama_chat(model: str, prompt: str) -> str:
    try:
        import ollama  # type: ignore
    except Exception as e:
        raise RuntimeError("Pacote 'ollama' não instalado. Instale: pip install ollama") from e

    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=OLLAMA_DETERMINISTIC_OPTIONS,
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def _cypher_is_read_only(cypher: str) -> bool:
    bad = [
        r"\bCREATE\b", r"\bMERGE\b", r"\bDELETE\b", r"\bSET\b",
        r"\bDROP\b", r"\bDETACH\b", r"\bREMOVE\b", r"\bLOAD\s+CSV\b",
        r"\bCALL\s+dbms\.", r"\bCALL\s+apoc\.", r"\bALTER\b",
    ]
    up = cypher.upper()
    for pat in bad:
        if re.search(pat, up):
            return False
    return True


def _records_to_json(records: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        try:
            row = dict(r)
            out.append({str(k): _to_jsonable(v) for k, v in row.items()})
        except Exception:
            out.append({"_raw": str(r)})
    return out


# -------------------------
# IA1: Planner
# -------------------------
def ia1_planner(question: str) -> Dict[str, Any]:
    prompt = (
        "Você é a IA 1 (PLANNER).\n"
        "Retorne um JSON com as chaves:\n"
        "- backend_mode: 'graph_only' | 'rag_only' | 'hybrid_graph_rag'\n"
        "- needs_query_graph: boolean\n"
        "- needs_query_rag: boolean\n"
        "- rag_plan: objeto ou null (opcional)\n\n"
        f"Pergunta: {question}\n"
        "Responda SOMENTE com JSON."
    )
    raw = _ollama_chat(MODEL_NAME_PLANNER, prompt)
    data = _safe_json_loads(raw)
    if not isinstance(data, dict):
        data = {
            "backend_mode": "graph_only",
            "needs_query_graph": True,
            "needs_query_rag": False,
            "rag_plan": None,
            "_raw": raw,
        }
    bm = str(data.get("backend_mode", "graph_only") or "graph_only").strip().lower()
    if bm not in ("graph_only", "rag_only", "hybrid_graph_rag"):
        bm = "graph_only"
    data["backend_mode"] = bm
    data.setdefault("needs_query_graph", bm in ("graph_only", "hybrid_graph_rag"))
    data.setdefault("needs_query_rag", bm in ("rag_only", "hybrid_graph_rag"))
    data.setdefault("rag_plan", None)
    return data


# -------------------------
# IA2: Cypher + (opcional) corretor
# -------------------------
def ia2_generate_cypher(question: str, planner: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prompt_ia2 = (
        "Você é a IA 2 (CYPHER).\n"
        "Gere um CYPHER SOMENTE LEITURA para Neo4j que responda a pergunta.\n"
        "Evite escrita (CREATE/MERGE/DELETE/SET/DROP).\n\n"
        f"AISchema (IA1): {_jd(planner)}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda APENAS com o Cypher."
    )
    cypher_raw = _ollama_chat(MODEL_NAME_CYPHER, prompt_ia2)

    cypher = (cypher_raw or "").strip()
    cypher = re.sub(r"^```[a-zA-Z]*\s*", "", cypher).strip()
    cypher = re.sub(r"```\s*$", "", cypher).strip()

    debug = {"prompt_ia2": prompt_ia2, "raw_ia2": cypher_raw}

    if MODEL_NAME_CYPHER_CORRECTOR:
        payload = (
            "Você é a IA 2.5 (CYPHER_CORRECTOR).\n"
            "Receba um cypher e devolva um cypher corrigido SOMENTE LEITURA.\n"
            "Devolva APENAS o cypher.\n\n"
            f"Pergunta: {question}\n\n"
            f"Cypher candidato:\n{cypher}\n"
        )
        try:
            cypher_corr = _ollama_chat(MODEL_NAME_CYPHER_CORRECTOR, payload)
            cypher_corr = re.sub(r"^```[a-zA-Z]*\s*", "", (cypher_corr or "")).strip()
            cypher_corr = re.sub(r"```\s*$", "", cypher_corr).strip()
            if cypher_corr:
                debug["payload_ia25"] = payload
                debug["raw_ia25"] = cypher_corr
                cypher = cypher_corr
        except Exception as e:
            debug["ia25_error"] = str(e)

    return cypher, debug


# -------------------------
# Neo4j
# -------------------------
def run_neo4j(cypher: str) -> Dict[str, Any]:
    if not _cypher_is_read_only(cypher):
        return {"error": "Cypher bloqueado (não é somente leitura)", "cypher": cypher, "records": []}

    driver = get_neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            res = session.run(cypher)
            records = list(res)
        return {"error": None, "cypher": cypher, "records": _records_to_json(records)}
    except Exception as e:
        return {"error": str(e), "cypher": cypher, "records": []}


# -------------------------
# RAG (opcional)
# -------------------------
def run_rag(question: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    col = get_chroma_collection()
    if col is None:
        return {"enabled": False, "error": "Chroma não configurado", "documents": [], "metadatas": []}

    try:
        embedder = get_embedder()
        q_emb = embedder.encode(question).tolist()
        results = col.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where or None,
        )
        docs = (results.get("documents") or [[]])[0] if results.get("documents") else []
        metas = (results.get("metadatas") or [[]])[0] if results.get("metadatas") else []
        return {
            "enabled": True,
            "error": None,
            "documents": docs,
            "metadatas": metas,
            "where": where,
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e),
            "documents": [],
            "metadatas": [],
            "where": where,
        }


# -------------------------
# IA3: Final answer (COM contexto do último turno apenas)
# -------------------------
def ia3_final_answer(
    question: str,
    planner: Dict[str, Any],
    neo4j_result: Dict[str, Any],
    rag_result: Dict[str, Any],
    last_turn: Optional[Dict[str, str]],
) -> str:
    contexto = ""
    if last_turn:
        contexto = (
            "Contexto (somente a última interação):\n"
            f"Pergunta anterior: {last_turn.get('pergunta','')}\n"
            f"Resposta anterior: {last_turn.get('resposta_final','')}\n\n"
        )

    prompt = (
        f"{contexto}"
        "Você é a IA FINAL (ANSWER).\n"
        "Use os resultados (Neo4j e/ou RAG) para responder.\n"
        "Se Neo4j vier com erro, explique e sugira como reformular a pergunta.\n\n"
        f"AISchema (IA1): {_jd(planner)}\n\n"
        f"Neo4j: {_jd(neo4j_result)}\n\n"
        f"RAG: {_jd(rag_result)}\n\n"
        f"Pergunta atual: {question}\n\n"
        "Responda em português, direto e com clareza."
    )
    return _ollama_chat(MODEL_NAME_ANSWER, prompt)


# -------------------------
# Função principal (a que o main chama)
# -------------------------
def generate_response(message: str, user_id: str, user_key: str) -> str:
    last_turn = get_last_turn_context(user_key)

    planner = ia1_planner(message)

    cypher = ""
    ia2_debug: Dict[str, Any] = {}
    neo4j_result: Dict[str, Any] = {"error": None, "cypher": "", "records": []}
    rag_result: Dict[str, Any] = {"enabled": False, "error": None, "documents": [], "metadatas": []}

    needs_graph = bool(planner.get("needs_query_graph"))
    needs_rag = bool(planner.get("needs_query_rag"))

    if needs_graph:
        cypher, ia2_debug = ia2_generate_cypher(message, planner)
        neo4j_result = run_neo4j(cypher)

    if needs_rag:
        rag_result = run_rag(message, n_results=5, where=None)

    final_answer = ia3_final_answer(
        question=message,
        planner=planner,
        neo4j_result=neo4j_result,
        rag_result=rag_result,
        last_turn=last_turn,
    )

    entry = {
        "ts": now_iso(),
        "user_id": str(user_id),
        "user_key": str(user_key),
        "pergunta": message,
        "ias": {
            "IA1_PLANNER": planner,
            "IA2_CYPHER": {"cypher": cypher, **ia2_debug},
            "IA3_FINAL": {"resposta": final_answer},
        },
        "neo4j": neo4j_result,
        "rag": rag_result,
        "resposta_final": final_answer,
    }
    append_entry(user_key, entry)

    return final_answer

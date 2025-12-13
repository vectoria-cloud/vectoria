"""
agt.py — substituto do antigo "assistants + threads" (OpenAI) pelo pipeline:

IA1 (PLANNER / schema JSON) -> IA2 (CYPHER) -> Neo4j -> (opcional) RAG normativo -> IA3 (ANSWER)

Uso esperado (compatível com seu Flask):
    from funcoes import agent   # ou import agt as agent
    agent.generate_response("pergunta", "wa_id", user_key="sisa_web")

Observações:
- Não há chaves hard-coded aqui. Configure via variáveis de ambiente.
- Se você não usa RAG normativo (Chroma), pode desligar com:
    export ENABLE_NORMATIVE_RAG=0
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import shelve
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dependências externas do seu stack
import ollama
from neo4j import GraphDatabase
import chromadb


# =============================================================================
# Paths / armazenamento
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
# Se este arquivo estiver em /funcoes, assumimos que o root é o pai de /funcoes.
# Caso contrário, o root é o próprio diretório do arquivo.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")).resolve() if os.environ.get("PROJECT_ROOT") else (
    THIS_DIR.parent if THIS_DIR.name.lower() in ("funcoes", "src", "app") else THIS_DIR
)
DATA_DIR = PROJECT_ROOT / "files" / "ia"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DB_PATH = str(DATA_DIR / "history_shelve")  # shelve cria múltiplos arquivos


# =============================================================================
# Config — modelos / Neo4j / RAG
# =============================================================================

# IA1 (Planner)
MODEL_NAME_SCHEMA = os.environ.get("MODEL_NAME_SCHEMA", "PLANNER")

# IA2 (Cypher)
MODEL_NAME_CYPHER = os.environ.get("MODEL_NAME_CYPHER", "CYPHER_GENERATOR")

# IA3 (Resposta final)
MODEL_NAME_ANSWER = os.environ.get("MODEL_NAME_ANSWER", "ANSWER")

# Embeddings para o índice de schema
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "mxbai-embed-large")

# Neo4j
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "zWF$yls*J;K:DtC3")

# >>> DETALHE PEDIDO: banco específico do Neo4j <<<
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "vectoria")

NEO4J_QUERY_TIMEOUT_SEC = int(os.environ.get("NEO4J_QUERY_TIMEOUT_SEC", "30"))

# RAG normativo (Chroma)
ENABLE_NORMATIVE_RAG = os.environ.get("ENABLE_NORMATIVE_RAG", "1").strip() not in ("0", "false", "False")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "arbopedia")

# Cloud vs local (fallback)
CHROMA_MODE = os.environ.get("CHROMA_MODE", "cloud").strip().lower()  # "cloud" | "local"
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE", "arbopedia")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / ".chroma"))


# Schema RAG (para IA2) — índice local (schema_vetorizado)
INDEX_DIR = Path(os.environ.get("SCHEMA_INDEX_DIR", str(PROJECT_ROOT / "schema_vetorizado")))
EMB_PATH = INDEX_DIR / "schema_emb.npy"
CHUNKS_PATH = INDEX_DIR / "schema_chunks.json"


# =============================================================================
# Locks / warm-up
# =============================================================================

_schema_llm_lock = threading.Lock()
_schema_llm_warmed_up = False

_cypher_lock = threading.Lock()
_cypher_warmed_up = False

_chroma_lock = threading.Lock()
_chroma_ready = False

_schema_index_lock = threading.Lock()


# =============================================================================
# Neo4j driver singleton
# =============================================================================

_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# =============================================================================
# Schema index globals
# =============================================================================

_SCHEMA_CHUNKS: List[dict] = []
_SCHEMA_EMB_MATRIX: Optional[np.ndarray] = None
_SCHEMA_INDEX_LOADED = False

# índices determinísticos
_SCHEMA_RELS: List[dict] = []
_SCHEMA_ENTITY_DESCR: Dict[str, str] = {}
_SCHEMA_FIELDS_BY_ENTITY: Dict[str, List[dict]] = defaultdict(list)
_SCHEMA_TIPOS_BY_ENTITY: Dict[str, List[dict]] = defaultdict(list)
_SCHEMA_IDX_KIND_ENTITY: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # (kind, entity) -> idxs


# =============================================================================
# Chroma globals (RAG normativo)
# =============================================================================

_chroma_client = None
_rag_collection = None


# =============================================================================
# Utilitários
# =============================================================================

def _truncate(text: Any, max_len: int = 150) -> str:
    if not text:
        return ""
    s = str(text).strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _embed_text(text: str) -> List[float]:
    return ollama.embeddings(model=EMBED_MODEL_NAME, prompt=text)["embedding"]


def _build_schema_maps() -> None:
    global _SCHEMA_RELS, _SCHEMA_ENTITY_DESCR, _SCHEMA_FIELDS_BY_ENTITY, _SCHEMA_TIPOS_BY_ENTITY, _SCHEMA_IDX_KIND_ENTITY

    _SCHEMA_RELS = []
    _SCHEMA_ENTITY_DESCR = {}
    _SCHEMA_FIELDS_BY_ENTITY = defaultdict(list)
    _SCHEMA_TIPOS_BY_ENTITY = defaultdict(list)
    _SCHEMA_IDX_KIND_ENTITY = defaultdict(list)

    for i, ch in enumerate(_SCHEMA_CHUNKS):
        ent = ch.get("entity")
        kind = ch.get("kind")
        if not ent or not kind:
            continue

        _SCHEMA_IDX_KIND_ENTITY[(kind, ent)].append(i)

        if kind == "entity":
            _SCHEMA_ENTITY_DESCR.setdefault(ent, ch.get("descr", ""))
        elif kind == "field":
            _SCHEMA_FIELDS_BY_ENTITY[ent].append(ch)
        elif kind == "tipo":
            _SCHEMA_TIPOS_BY_ENTITY[ent].append(ch)
        elif kind == "rel":
            _SCHEMA_RELS.append(ch)


def _load_schema_embedding_index() -> bool:
    global _SCHEMA_CHUNKS, _SCHEMA_EMB_MATRIX, _SCHEMA_INDEX_LOADED
    if not EMB_PATH.exists() or not CHUNKS_PATH.exists():
        _SCHEMA_INDEX_LOADED = False
        return False

    try:
        _SCHEMA_EMB_MATRIX = np.load(str(EMB_PATH))
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            _SCHEMA_CHUNKS = json.load(f)

        _build_schema_maps()
        _SCHEMA_INDEX_LOADED = True
        return True
    except Exception:
        _SCHEMA_INDEX_LOADED = False
        return False


def _ensure_schema_index_loaded() -> None:
    global _SCHEMA_INDEX_LOADED
    if _SCHEMA_INDEX_LOADED:
        return
    with _schema_index_lock:
        if not _SCHEMA_INDEX_LOADED:
            _load_schema_embedding_index()


def search_schema_chunks(query: str, allowed_entities: Optional[List[str]] = None, top_k: int = 50) -> List[dict]:
    _ensure_schema_index_loaded()
    if _SCHEMA_EMB_MATRIX is None or not _SCHEMA_CHUNKS:
        return []

    q_emb = np.array(_embed_text(query), dtype="float32")
    q_emb /= (np.linalg.norm(q_emb) + 1e-9)

    emb_norm = _SCHEMA_EMB_MATRIX / (np.linalg.norm(_SCHEMA_EMB_MATRIX, axis=1, keepdims=True) + 1e-9)
    sims = emb_norm @ q_emb

    idxs = np.arange(len(_SCHEMA_CHUNKS))
    if allowed_entities:
        allowed = set(allowed_entities)
        mask = np.array([c.get("entity") in allowed for c in _SCHEMA_CHUNKS])
        if np.any(mask):
            idxs = idxs[mask]
            sims = sims[mask]

    if sims.size == 0:
        return []

    k_real = min(top_k, sims.shape[0])
    top_idx = np.argpartition(-sims, k_real - 1)[:k_real]
    sorted_idx = top_idx[np.argsort(-sims[top_idx])]

    return [_SCHEMA_CHUNKS[int(idxs[i])] for i in sorted_idx]


def get_relations_among_entities(target_entities: List[str]) -> List[dict]:
    _ensure_schema_index_loaded()
    ents = set(target_entities or [])
    if not ents:
        return []

    seen = set()
    rels = []
    for ch in _SCHEMA_RELS:
        de = ch.get("de")
        para = ch.get("para")
        tipo_rel = ch.get("tipo_rel")
        if de in ents and para in ents and tipo_rel:
            key = (de, tipo_rel, para)
            if key not in seen:
                seen.add(key)
                rels.append(ch)
    return rels


def rank_tipos(entity: str, query: str, top_k: int = 5) -> List[dict]:
    _ensure_schema_index_loaded()
    idxs = _SCHEMA_IDX_KIND_ENTITY.get(("tipo", entity), [])
    if not idxs:
        return []
    if _SCHEMA_EMB_MATRIX is None:
        return (_SCHEMA_TIPOS_BY_ENTITY.get(entity, []) or [])[:top_k]

    q_emb = np.array(_embed_text(query), dtype="float32")
    q_emb /= (np.linalg.norm(q_emb) + 1e-9)

    M = _SCHEMA_EMB_MATRIX[idxs]
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sims = M @ q_emb

    k_real = min(top_k, sims.shape[0])
    top_idx_local = np.argpartition(-sims, k_real - 1)[:k_real]
    sorted_local = top_idx_local[np.argsort(-sims[top_idx_local])]
    return [_SCHEMA_CHUNKS[idxs[i]] for i in sorted_local]


def ensure_keyword_tipos(tipos: List[dict], keyword_fields: set) -> List[dict]:
    chosen = {t.get("tipo_codigo") for t in (tipos or [])}
    all_t = _SCHEMA_TIPOS_BY_ENTITY.get("AtividadeExec", []) or []
    for t in all_t:
        code = t.get("tipo_codigo")
        if code in chosen:
            continue
        campos = (t.get("campos_exclusivos") or []) + (t.get("campos_comuns") or [])
        if any(c in keyword_fields for c in campos):
            tipos.append(t)
            chosen.add(code)
    return tipos


def build_rag_context(aischema: dict, question: str) -> str:
    """
    Monta contexto "schema-first" (para IA2 gerar Cypher), seguindo a lógica do seu teste.py.
    Se targets.entities não vier do PLANNER, faz fallback por busca vetorial no schema.
    """
    _ensure_schema_index_loaded()

    entities = (aischema.get("targets", {}) or {}).get("entities") or []

    # ------------------------------
    # Fallback antigo (sem targets)
    # ------------------------------
    if not entities:
        chunks = search_schema_chunks(query=question, allowed_entities=None, top_k=60)
        if not chunks:
            return "Nenhum contexto encontrado."

        grouped: Dict[str, Dict[str, Any]] = {}
        global_rels = set()

        for ch in chunks:
            ent = ch.get("entity", "Geral")
            kind = ch.get("kind")

            if kind in ["entity", "field", "tipo"]:
                g = grouped.setdefault(ent, {"descr": "", "fields": [], "tipos": []})
                if kind == "entity":
                    if not g["descr"]:
                        g["descr"] = ch.get("descr", "")
                elif kind == "field":
                    names = [f.get("name") for f in g["fields"]]
                    if ch.get("name") not in names:
                        g["fields"].append(ch)
                elif kind == "tipo":
                    codes = [t.get("tipo_codigo") for t in g["tipos"]]
                    if ch.get("tipo_codigo") not in codes:
                        g["tipos"].append(ch)

            elif kind == "rel":
                rel_str = f"({ch.get('de')})-[:{ch.get('tipo_rel')}]->({ch.get('para')})"
                global_rels.add(rel_str)

        lines: List[str] = []
        lines.append("### ESTRUTURA (Backbone):")
        lines.append("   - (Municipio)-[:TEM_DADO_NO_DIA]->(Dia)")
        lines.append("")

        for ent, info in grouped.items():
            lines.append(f"### Entidade: {ent}")
            if info.get("descr"):
                lines.append(f"   DESCRIÇÃO: {_truncate(info['descr'], 200)}")

            fields = info.get("fields") or []
            if fields:
                lines.append("   PROPRIEDADES:")
                for f in sorted(fields, key=lambda x: (x.get("name") or "")):
                    tipo = f"({f.get('tipo', 'str')})"
                    desc = f" - {_truncate(f.get('descr', ''), 200)}"
                    lines.append(f"     - {f.get('name')} {tipo}{desc}")

            # Tipos SISAWEB quando aparecerem no fallback
            if ent == "AtividadeExec" and (info.get("tipos") or []):
                lines.append("   TIPOS SISAWEB (Use: WHERE ae.sisaweb_tipo = ID):")
                for t in info["tipos"]:
                    tid = t.get("tipo_codigo")
                    name = t.get("name", f"tipo {tid}")
                    desc = _truncate(t.get("descr", ""), 220)
                    lines.append(f"     - {name} -> sisaweb_tipo: {tid}")
                    lines.append(f"       Desc: {desc}")

            lines.append("")

        if global_rels:
            lines.append("### RELACIONAMENTOS (Encontrados no fallback):")
            for r in sorted(global_rels):
                lines.append(f"   - {r}")

        return "\n".join(lines)

    # ------------------------------
    # Caminho principal (com targets)
    # ------------------------------
    lines: List[str] = []
    lines.append("### ESTRUTURA (Backbone):")
    lines.append("   - (Municipio)-[:TEM_DADO_NO_DIA]->(Dia)")
    lines.append("")

    for ent in entities:
        lines.append(f"### Entidade: {ent}")

        descr = _SCHEMA_ENTITY_DESCR.get(ent, "")
        if descr:
            lines.append(f"   DESCRIÇÃO: {_truncate(descr, 200)}")

        fields = _SCHEMA_FIELDS_BY_ENTITY.get(ent, []) or []
        if fields:
            lines.append("   PROPRIEDADES:")
            for f in sorted(fields, key=lambda x: x.get("name", "")):
                fname = f.get("name")
                ftype = f.get("tipo", "str")
                fdesc = _truncate(f.get("descr", ""), 200)
                lines.append(f"     - {fname} ({ftype}) - {fdesc}")

        # AtividadeExec -> selecionar melhores TIPOS p/ a pergunta
        if ent == "AtividadeExec":
            tipos_ranked = rank_tipos("AtividadeExec", question, top_k=5)

            qlow = (question or "").lower()

            # reforço por palavras-chave comuns
            if "nebul" in qlow or "ubv" in qlow:
                tipos_ranked = ensure_keyword_tipos(
                    tipos_ranked,
                    keyword_fields={"nebulizacao", "nebul", "neb"},
                )

            if "larv" in qlow or "liraa" in qlow or "adl" in qlow:
                tipos_ranked = ensure_keyword_tipos(
                    tipos_ranked,
                    keyword_fields={"ib_larva", "ip_larva", "im_larva", "rec_larva"},
                )

            if tipos_ranked:
                lines.append("   TIPOS SISAWEB (Use: WHERE ae.sisaweb_tipo = ID):")

                def get_code(t):
                    try:
                        return int(t.get("tipo_codigo", 0))
                    except Exception:
                        return 999

                for t in sorted(tipos_ranked, key=get_code):
                    tid = t.get("tipo_codigo")
                    name = t.get("name", f"tipo {tid}")
                    desc = _truncate(t.get("descr", ""), 220)

                    campos_ex = t.get("campos_exclusivos") or []
                    campos_co = t.get("campos_comuns") or []

                    campos_all = []
                    for c in (campos_ex + campos_co):
                        if c not in campos_all:
                            campos_all.append(c)

                    lines.append(f"     - {name} -> sisaweb_tipo: {tid}")
                    lines.append(f"       Desc: {desc}")
                    if campos_all:
                        lines.append(f"       CAMPOS DO TIPO: {', '.join(campos_all)}")

        lines.append("")

    # Relações entre as entidades alvo do plano
    rels = get_relations_among_entities(entities)
    if rels:
        lines.append("### RELACIONAMENTOS (Entre Entidades do PLANO):")
        for ch in sorted(rels, key=lambda r: (r.get("de", ""), r.get("tipo_rel", ""), r.get("para", ""))):
            lines.append(f"   - ({ch.get('de')})-[:{ch.get('tipo_rel')}]->({ch.get('para')})")

    return "\n".join(lines)


# =============================================================================
# IA 1 — Planner (schema JSON)
# =============================================================================

def _warm_up_schema_llm() -> None:
    global _schema_llm_warmed_up
    if _schema_llm_warmed_up:
        return
    try:
        ollama.chat(model=MODEL_NAME_SCHEMA, messages=[{"role": "user", "content": "ping"}])
        _schema_llm_warmed_up = True
    except Exception as e:
        print(f"[WARN] Falha no warm-up {MODEL_NAME_SCHEMA}: {e}")


def get_ai_schema_dict(question: str) -> dict:
    with _schema_llm_lock:
        _warm_up_schema_llm()

        start_time = time.time()
        try:
            resp = ollama.chat(
                model=MODEL_NAME_SCHEMA,
                messages=[{"role": "user", "content": question}],
                format="json",
            )

            content = resp.get("message", {}).get("content", "")

            if isinstance(content, str):
                data = json.loads(content)
            elif isinstance(content, dict):
                data = content
            else:
                data = {}

            if not isinstance(data, dict):
                data = {}

            data.setdefault("intent", "lookup")
            data.setdefault("needs_query", False)

            data.setdefault("backend_mode", "graphs")  # graphs | rag | hybrid
            data.setdefault("needs_query_rag", False)
            data.setdefault("rag_plan", {})
            data.setdefault("graph_plan", {})
            data.setdefault("hybrid_plan", {})

            data["_generation_time"] = time.time() - start_time
            return data

        except json.JSONDecodeError:
            # fallback se JSON vier quebrado
            return {
                "intent": "lookup",
                "needs_query": False,
                "backend_mode": "graphs",
                "needs_query_rag": False,
                "rag_plan": {},
                "graph_plan": {},
                "hybrid_plan": {},
                "reason": "Erro na geração do schema JSON.",
                "_generation_time": time.time() - start_time,
            }
        except Exception as e:
            print(f"[ERRO] Erro na chamada Ollama (Planner): {e}")
            return {"_generation_time": time.time() - start_time}


# =============================================================================
# IA 2 — Cypher
# =============================================================================

def _warm_up_cypher() -> None:
    global _cypher_warmed_up
    if _cypher_warmed_up:
        return
    try:
        ollama.chat(model=MODEL_NAME_CYPHER, messages=[{"role": "user", "content": "ping"}])
    except Exception:
        pass
    _cypher_warmed_up = True


def clean_cypher_output(text: str) -> str:
    match = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(MATCH|WITH|CALL)\s.*", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return (text or "").strip()


def cypher_llm_invoke(prompt: str) -> str:
    with _cypher_lock:
        _warm_up_cypher()
        resp = ollama.chat(
            model=MODEL_NAME_CYPHER,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 8192},
            stream=False,
        )
    return clean_cypher_output(resp["message"]["content"])


def build_cypher_prompt(question: str, aischema: dict) -> str:
    rag_context = build_rag_context(aischema, question)
    aischema_json = json.dumps(aischema, ensure_ascii=False, indent=2)
    prompt = f"""
PERGUNTA: "{question}"

PLANO (JSON):
{aischema_json}

CONTEXTO DO BANCO (RAG):
{rag_context}
""".strip()
    return prompt


# =============================================================================
# Neo4j
# =============================================================================

def run_cypher_in_neo4j(cypher: str) -> List[dict]:
    if not cypher or not cypher.strip():
        return []
    with _driver.session(database=NEO4J_DATABASE) as session:
        return session.run(cypher, timeout=NEO4J_QUERY_TIMEOUT_SEC).data()


# =============================================================================
# RAG normativo (Chroma)
# =============================================================================

def _init_chroma_if_needed() -> None:
    global _chroma_ready, _chroma_client, _rag_collection
    if _chroma_ready or not ENABLE_NORMATIVE_RAG:
        return

    with _chroma_lock:
        if _chroma_ready or not ENABLE_NORMATIVE_RAG:
            return

        try:
            if CHROMA_MODE == "cloud":
                if not CHROMA_API_KEY or not CHROMA_TENANT:
                    raise RuntimeError(
                        "CHROMA_MODE=cloud mas CHROMA_API_KEY/CHROMA_TENANT não estão definidos."
                    )
                _chroma_client = chromadb.CloudClient(
                    api_key=CHROMA_API_KEY,
                    tenant=CHROMA_TENANT,
                    database=CHROMA_DATABASE,
                )
            else:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

            _rag_collection = _chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            _chroma_ready = True
        except Exception as e:
            # Não mata o pipeline — só desliga RAG normativo
            print(f"[WARN] Falha ao inicializar Chroma ({CHROMA_MODE}): {e}")
            _chroma_ready = False
            _chroma_client = None
            _rag_collection = None


def retrieve_rag_context(question: str, aischema: dict, default_top_k: int = 8) -> List[dict]:
    """
    Busca trechos normativos na coleção Chroma (arbopedia), usando rag_plan.
    Retorna lista de chunks com metadados principais.
    """
    if not ENABLE_NORMATIVE_RAG:
        return []

    _init_chroma_if_needed()
    if _rag_collection is None:
        return []

    rag_plan = aischema.get("rag_plan") or {}

    query_text = rag_plan.get("query_text") or rag_plan.get("query") or rag_plan.get("rag_query") or question
    top_k = int(rag_plan.get("top_k") or default_top_k)
    where = rag_plan.get("filters") or rag_plan.get("doc_filters") or {}

    res = _rag_collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=where or None,
        include=["documents", "metadatas", "ids"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    chunks = []
    for i, doc in enumerate(docs):
        meta = metas[i] or {}
        chunks.append(
            {
                "id": ids[i] if i < len(ids) else None,
                "text": doc,
                "source": meta.get("pdf") or meta.get("source_pdf") or meta.get("filename"),
                "page": meta.get("page") or meta.get("page_number"),
                "doc_id": meta.get("doc_id"),
                "doc_title": meta.get("doc_title"),
                "doc_type": meta.get("doc_type"),
                "diseases_raw": meta.get("diseases"),
                "themes_raw": meta.get("themes"),
                "metadata": meta,
            }
        )
    return chunks


# =============================================================================
# IA 3 — Resposta final (ANSWER)
# =============================================================================

def format_graph_context_for_answer(graph_result: Optional[List[dict]]) -> str:
    if not graph_result:
        return "Nenhum dado de grafo foi utilizado para esta pergunta."

    linhas = [str(row) for row in graph_result[:10]]
    if len(graph_result) > 10:
        linhas.append(f"... (+{len(graph_result) - 10} linhas adicionais)")

    return "Principais dados retornados pelo grafo:\n" + "\n".join(linhas)


def format_rag_context_for_answer(chunks: Optional[List[dict]], max_chars: int = 4000) -> str:
    if not chunks:
        return "Nenhum trecho de documento normativo foi recuperado."

    parts: List[str] = []
    total = 0
    for ch in chunks:
        src = ch.get("source") or "desconhecido"
        page = ch.get("page")
        header = f"[{src}, pág. {page}]\n" if page is not None else f"[{src}]\n"
        block = header + (ch.get("text") or "").strip() + "\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def answer_with_graph_and_rag(question: str, aischema: dict, graph_result: List[dict], rag_chunks: List[dict]) -> str:
    backend_mode = aischema.get("backend_mode")
    _ = aischema.get("intent")

    graph_context = format_graph_context_for_answer(graph_result)
    rag_context = format_rag_context_for_answer(rag_chunks)

    aischema_str = json.dumps(
        {
            "intent": aischema.get("intent"),
            "backend_mode": backend_mode,
            "targets": aischema.get("targets"),
            "filters": aischema.get("filters"),
            "metrics": aischema.get("metrics"),
        },
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = f"""
Pergunta original do usuário:
{question}

Resumo do plano (AISchema) escolhido pelo Planner:
{aischema_str}

=== CONTEXTO DE GRAFO ===
{graph_context}

=== TRECHOS DE DOCUMENTOS NORMATIVOS (RAG) ===
{rag_context}

Com base APENAS nessas informações, produza uma resposta em português,
clara e acionável, adequada para profissionais de saúde, vigilância e gestão.
Deixe claro, quando fizer sentido, o que vem dos dados do grafo e o que vem dos documentos normativos.
""".strip()

    resp = ollama.chat(
        model=MODEL_NAME_ANSWER,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp["message"]["content"].strip()


# =============================================================================
# Orquestrador IA1 + IA2 + Neo4j + RAG + IA3
# =============================================================================

def ask_arbo_agent(question: str) -> dict:
    timers: Dict[str, float] = {}

    # --- IA 1: Planner ---
    t0 = time.time()
    aischema = get_ai_schema_dict(question) or {}
    timers["ia1_planner"] = aischema.pop("_generation_time", time.time() - t0)

    backend_mode = aischema.get("backend_mode", "graph_only")
    needs_query_graph = bool(aischema.get("needs_query", False))
    needs_query_rag = bool(aischema.get("needs_query_rag", False))

    cypher = ""
    graph_result: List[dict] = []
    neo4j_error: Optional[str] = None
    rag_chunks: List[dict] = []

    # --- IA 2 + Neo4j ---
    if needs_query_graph:
        t1 = time.time()
        prompt_cypher = build_cypher_prompt(question, aischema)
        cypher = cypher_llm_invoke(prompt_cypher)
        timers["ia2_cypher"] = time.time() - t1

        if cypher and cypher.strip():
            t2 = time.time()
            try:
                graph_result = run_cypher_in_neo4j(cypher)
            except Exception as e:
                neo4j_error = str(e)
            timers["neo4j_exec"] = time.time() - t2

    # --- RAG normativo ---
    if needs_query_rag:
        t3 = time.time()
        rag_chunks = retrieve_rag_context(question, aischema)
        timers["rag_retrieval"] = time.time() - t3

    # --- IA 3 ---
    t4 = time.time()
    answer = answer_with_graph_and_rag(question, aischema, graph_result, rag_chunks)
    timers["ia3_answer"] = time.time() - t4

    return {
        "question": question,
        "backend_mode": backend_mode,
        "planner": aischema,
        "graph": {
            "needs_graph": needs_query_graph,
            "planner_graph_plan": aischema.get("graph_plan"),
            "cypher": cypher,
            "result_rows": graph_result,
            "error": neo4j_error,
        },
        "rag": {
            "needs_rag": needs_query_rag,
            "planner_rag_plan": aischema.get("rag_plan"),
            "retrieved_chunks": rag_chunks,
        },
        "final_answer": answer,
        "timers": timers,
    }


# =============================================================================
# "Interface" compatível com seu Flask (substitui o agt antigo)
# =============================================================================

MAX_HISTORY_TURNS = int(os.environ.get("AGT_MAX_HISTORY_TURNS", "6"))
MAX_HISTORY_CHARS = int(os.environ.get("AGT_MAX_HISTORY_CHARS", "3000"))
USE_MEMORY = os.environ.get("AGT_USE_MEMORY", "1").strip() not in ("0", "false", "False")


def _history_key(wa_id: str, user_key: Optional[str]) -> str:
    user_key = (user_key or "").strip() or "default"
    return f"{user_key}::{wa_id}"


def _load_history(key: str) -> List[dict]:
    try:
        with shelve.open(HISTORY_DB_PATH) as db:
            return list(db.get(key, []))
    except Exception:
        return []


def _save_history(key: str, history: List[dict]) -> None:
    try:
        with shelve.open(HISTORY_DB_PATH, writeback=True) as db:
            db[key] = history
    except Exception:
        pass


def _build_question_with_history(message_body: str, history: List[dict]) -> str:
    if not history:
        return message_body

    # pega só os últimos N turns
    turns = history[-MAX_HISTORY_TURNS:]
    parts = []
    for t in turns:
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if u:
            parts.append(f"Usuário: {u}")
        if a:
            parts.append(f"Assistente: {a}")

    ctx = "\n".join(parts)
    if len(ctx) > MAX_HISTORY_CHARS:
        ctx = ctx[-MAX_HISTORY_CHARS:]

    return f"""Histórico recente (para contexto):
{ctx}

Mensagem atual do usuário:
{message_body}
""".strip()


def generate_response(message_body: str, wa_id: str, user_key: Optional[str] = None, return_full: bool = False):
    """
    Drop-in replacement:
      - retorna string (resposta final) por padrão
      - se return_full=True, retorna o pipeline_output completo (dict)
    """
    message_body = (message_body or "").strip()
    if not message_body:
        return "" if not return_full else {"final_answer": ""}

    key = _history_key(str(wa_id), user_key)

    history = _load_history(key) if USE_MEMORY else []
    question = _build_question_with_history(message_body, history) if USE_MEMORY else message_body

    pipeline_output = ask_arbo_agent(question)
    answer = (pipeline_output.get("final_answer") or "").strip()

    if USE_MEMORY:
        history.append({"ts": time.time(), "user": message_body, "assistant": answer})
        # corta histórico para não crescer sem limites
        if len(history) > 50:
            history = history[-50:]
        _save_history(key, history)

    return pipeline_output if return_full else answer

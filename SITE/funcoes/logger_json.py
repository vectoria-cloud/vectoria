"""Logger simples em JSON para histórico por usuário.

Salva em: files/json/<user_key>.json

Estrutura:
  - o arquivo é uma LISTA de entradas (cada entrada = 1 interação)
  - cada entrada guarda: pergunta, respostas de cada IA, cypher, retorno neo4j, retorno rag, resposta_final
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_lock = Lock()


def _base_dir() -> Path:
    return Path("files") / "json"


def _path_for(user_key: str) -> Path:
    return _base_dir() / f"{user_key}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history(user_key: str) -> List[Dict[str, Any]]:
    p = _path_for(user_key)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_last_turn_context(user_key: str) -> Optional[Dict[str, str]]:
    """Retorna somente o ÚLTIMO turno (pergunta + resposta_final)."""
    hist = load_history(user_key)
    if not hist:
        return None
    last = hist[-1]
    pergunta = str(last.get("pergunta", "") or "")
    resposta = str(last.get("resposta_final", "") or "")
    if not pergunta and not resposta:
        return None
    return {"pergunta": pergunta, "resposta_final": resposta}


def append_entry(user_key: str, entry: Dict[str, Any]) -> None:
    """Append com escrita atômica (tmp + replace)."""
    base = _base_dir()
    base.mkdir(parents=True, exist_ok=True)
    p = _path_for(user_key)

    with _lock:
        hist = load_history(user_key)
        hist.append(entry)

        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2, default=str)

        os.replace(tmp, p)

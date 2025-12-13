from __future__ import annotations

import os
import shelve
from pathlib import Path
from typing import Dict, List, Literal

Role = Literal["user", "assistant"]

IA_DIR = Path(os.getenv("IA_DIR", "files/ia"))
IA_DIR.mkdir(parents=True, exist_ok=True)

_CONV_DB_PATH = str(IA_DIR / "arbopedia_conversations_db")


def get_history(wa_id: str, max_turns: int = 6) -> List[Dict[str, str]]:
    """Retorna o histórico recente (pares user/assistant)."""
    with shelve.open(_CONV_DB_PATH) as db:
        hist = db.get(str(wa_id), [])
    # cada turno tem 2 mensagens (user+assistant)
    return list(hist)[-(max_turns * 2):]


def append_history(wa_id: str, role: Role, content: str, max_keep_turns: int = 30) -> None:
    """Acrescenta uma mensagem ao histórico e limita o tamanho."""
    wa_id = str(wa_id)
    msg = {"role": role, "content": str(content or "")}

    with shelve.open(_CONV_DB_PATH, writeback=True) as db:
        hist = list(db.get(wa_id, []))
        hist.append(msg)

        # limita o tamanho total salvo
        max_msgs = max_keep_turns * 2
        if len(hist) > max_msgs:
            hist = hist[-max_msgs:]

        db[wa_id] = hist


def inject_history_as_text(question: str, history: List[Dict[str, str]]) -> str:
    """Formata histórico como texto (apenas para log/trace)."""
    if not history:
        return question

    chunks = ["Contexto da conversa (últimas mensagens):"]
    for m in history:
        role = "Usuário" if m.get("role") == "user" else "Assistente"
        chunks.append(f"{role}: {m.get('content','')}".strip())
    chunks.append("")
    chunks.append("Pergunta atual:")
    chunks.append(question)
    return "\n".join(chunks).strip()

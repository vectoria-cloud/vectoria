from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

JsonLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_jsonable(obj: Any, *, max_chars: int = 20000) -> JsonLike:
    try:
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return obj if len(obj) <= max_chars else obj[:max_chars] + "...(truncado)"
        if isinstance(obj, dict):
            return {str(k): _safe_jsonable(v, max_chars=max_chars) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe_jsonable(v, max_chars=max_chars) for v in obj]
        json.dumps(obj)
        return obj  
    except Exception:
        s = str(obj)
        return s if len(s) <= max_chars else s[:max_chars] + "...(truncado)"


@dataclass
class Step:
    name: str
    at: str
    model: Optional[str] = None
    input: JsonLike = None
    output: JsonLike = None
    meta: JsonLike = None


class TraceLogger:
    def __init__(self, user_key: str, *, base_dir: str = "files/json", max_chars: int = 20000):
        self.user_key = str(user_key or "default")
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs" / self.user_key
        self.history_path = self.base_dir / f"{self.user_key}.json"
        self.max_chars = max_chars

        self._history_lock = threading.Lock()
        self._runs_lock = threading.Lock()
        self._active: Dict[str, Dict[str, Any]] = {}

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_history_list(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _append_history_item(self, item: Dict[str, Any]) -> None:
        with self._history_lock:
            hist = self._load_history_list()
            hist.append(item)
            self._atomic_write_json(self.history_path, hist)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _flush_run(self, run_id: str) -> None:
        data = self._active.get(run_id)
        if data is None:
            return
        self._atomic_write_json(self._run_path(run_id), data)

    def start_run(
        self,
        *,
        question: str,
        user_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        question_with_context: Optional[str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        now = _utc_now_iso()

        run = {
            "run_id": run_id,
            "user_key": self.user_key,
            "user_id": user_id,
            "started_at": now,
            "ended_at": None,
            "status": "running",
            "question": question,
            "question_with_context": question_with_context,
            "meta": _safe_jsonable(meta or {}, max_chars=self.max_chars),
            "steps": [],
            "final_answer": None,
            "error": None,
        }

        with self._runs_lock:
            self._active[run_id] = run
            self._flush_run(run_id)

        self._append_history_item({
            "emissor": self.user_key,
            "pergunta": question,
            "run_id": run_id,
            "at": now,
        })

        return run_id

    def log_step(
        self,
        run_id: str,
        *,
        name: str,
        model: Optional[str] = None,
        input: Any = None,
        output: Any = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        step = {
            "name": name,
            "at": _utc_now_iso(),
            "model": model,
            "input": _safe_jsonable(input, max_chars=self.max_chars),
            "output": _safe_jsonable(output, max_chars=self.max_chars),
            "meta": _safe_jsonable(meta or {}, max_chars=self.max_chars),
        }

        with self._runs_lock:
            run = self._active.get(run_id)
            if not run:
                path = self._run_path(run_id)
                if path.exists():
                    run = json.loads(path.read_text(encoding="utf-8"))
                else:
                    return
                self._active[run_id] = run

            run_steps = run.get("steps") or []
            run_steps.append(step)
            run["steps"] = run_steps
            self._flush_run(run_id)

    def finalize_run(self, run_id: str, *, final_answer: str, status: str = "ok", extra: Optional[Dict[str, Any]] = None) -> None:
        now = _utc_now_iso()
        with self._runs_lock:
            run = self._active.get(run_id)
            if not run:
                path = self._run_path(run_id)
                if path.exists():
                    run = json.loads(path.read_text(encoding="utf-8"))
                else:
                    return
                self._active[run_id] = run

            run["final_answer"] = final_answer
            run["ended_at"] = now
            run["status"] = status
            if extra:
                meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
                if not isinstance(meta, dict):
                    meta = {}
                meta.update(_safe_jsonable(extra, max_chars=self.max_chars) or {})
                run["meta"] = meta
            self._flush_run(run_id)

        self._append_history_item({
            "emissor": "openai",  
            "resposta": final_answer,
            "run_id": run_id,
            "at": now,
        })

    def error_run(self, run_id: str, *, error: Any) -> None:
        now = _utc_now_iso()
        with self._runs_lock:
            run = self._active.get(run_id)
            if not run:
                path = self._run_path(run_id)
                if path.exists():
                    run = json.loads(path.read_text(encoding="utf-8"))
                else:
                    return
                self._active[run_id] = run

            run["error"] = _safe_jsonable(error, max_chars=self.max_chars)
            run["ended_at"] = now
            run["status"] = "error"
            self._flush_run(run_id)

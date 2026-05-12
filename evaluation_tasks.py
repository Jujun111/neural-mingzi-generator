import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


SUPPORTED_TASK_TYPES = ("blind_pairwise", "dynasty_guess")


class EvaluationTaskPool:
    def __init__(self, pool_path: Optional[Path] = None):
        self.pool_path = pool_path
        self._payload: Dict[str, Any] = {"task_pools": {}}
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._detail = "Evaluation task pool is unavailable."
        if pool_path is not None:
            self.load(pool_path)

    def load(self, pool_path: Path) -> None:
        self.pool_path = pool_path
        if not pool_path.exists():
            self._payload = {"task_pools": {}}
            self._by_id = {}
            self._detail = f"Evaluation task pool not found at {pool_path}."
            return

        payload = json.loads(pool_path.read_text(encoding="utf-8"))
        task_pools = payload.get("task_pools", {})
        if not isinstance(task_pools, dict):
            raise ValueError("Task pool payload must contain a 'task_pools' object.")

        self._payload = payload
        self._by_id = {}
        for task_type, tasks in task_pools.items():
            if task_type not in SUPPORTED_TASK_TYPES or not isinstance(tasks, list):
                continue
            for task in tasks:
                task_id = task.get("task_id")
                if isinstance(task_id, str):
                    self._by_id[task_id] = task

        self._detail = f"Loaded evaluation task pool from {pool_path}."

    def is_available(self) -> bool:
        return any(self._payload.get("task_pools", {}).get(task_type) for task_type in SUPPORTED_TASK_TYPES)

    def detail(self) -> str:
        return self._detail

    def list_task_types(self) -> Dict[str, int]:
        return {
            task_type: len(self._payload.get("task_pools", {}).get(task_type, []))
            for task_type in SUPPORTED_TASK_TYPES
        }

    def get_task(self, task_type: str) -> Dict[str, Any]:
        tasks = self._payload.get("task_pools", {}).get(task_type, [])
        if task_type not in SUPPORTED_TASK_TYPES:
            raise KeyError(f"Unsupported task type '{task_type}'.")
        if not tasks:
            raise KeyError(f"No tasks available for '{task_type}'.")
        task = random.choice(tasks)
        return self._public_task(task)

    def get_task_metadata(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._by_id:
            raise KeyError(f"Unknown task id '{task_id}'.")
        return self._by_id[task_id]

    def _public_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "prompt_type": task.get("prompt_type"),
            "title": task.get("title"),
            "prompt": task.get("prompt"),
            "presented_payload": task.get("presented_payload", {}),
            "response_options": task.get("response_options", []),
        }

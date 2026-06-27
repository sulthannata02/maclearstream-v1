"""
Experiment Tracker — Built-in experiment logging.

Setiap training run otomatis di-log ke folder ``experiments/``
sebagai file JSON. Bisa di-compare dan di-replay.

Contoh:
    >>> tracker = ExperimentTracker()
    >>> with tracker.start_run("RF-v2") as run:
    ...     run.log_params({"n_estimators": 300})
    ...     run.log_metrics({"accuracy": 0.95})
    ...     run.log_model("random_forest")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from config.settings import get_paths_config


# ══════════════════════════════════════════════
# Experiment Run
# ══════════════════════════════════════════════

class ExperimentRun:
    """
    Representasi satu kali training run.

    Menyimpan parameter, metrik, nama model, dan metadata waktu.
    """

    def __init__(
        self,
        name: str,
        experiments_dir: str,
    ) -> None:
        self.run_id: str = uuid4().hex[:8]
        self.name: str = name
        self.experiments_dir: str = experiments_dir
        self.start_time: str = datetime.now().isoformat()
        self.end_time: str | None = None
        self.execution_duration: float | None = None
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.model_name: str = ""
        self.tags: dict[str, str] = {}
        self.status: str = "running"

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrik evaluasi."""
        self.metrics.update(metrics)

    def log_model(self, model_name: str) -> None:
        """Log nama model yang digunakan."""
        self.model_name = model_name

    def log_tags(self, tags: dict[str, str]) -> None:
        """Log tags/label tambahan."""
        self.tags.update(tags)

    def to_dict(self) -> dict[str, Any]:
        """Konversi ke dict untuk serialisasi."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "model_name": self.model_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_duration": self.execution_duration,
            "status": self.status,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
        }

    def save(self) -> Path:
        """Simpan run ke file JSON."""
        dir_path = Path(self.experiments_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{self.run_id}_{self.name}.json"
        filepath = dir_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = datetime.now().isoformat()
        try:
            dt_start = datetime.fromisoformat(self.start_time)
            dt_end = datetime.fromisoformat(self.end_time)
            self.execution_duration = (dt_end - dt_start).total_seconds()
        except Exception:
            self.execution_duration = 0.0
        self.status = "failed" if exc_type else "completed"
        self.save()



# ══════════════════════════════════════════════
# Experiment Tracker
# ══════════════════════════════════════════════

class ExperimentTracker:
    """
    Manager untuk experiment runs.

    Menyediakan API untuk start run, list runs, dan compare.
    """

    def __init__(self, experiments_dir: str | None = None) -> None:
        if experiments_dir:
            self.experiments_dir = experiments_dir
        else:
            paths = get_paths_config()
            self.experiments_dir = paths.get("experiments_dir", "experiments")

    def start_run(self, name: str = "unnamed") -> ExperimentRun:
        """
        Mulai experiment run baru.

        Gunakan sebagai context manager:
            >>> with tracker.start_run("my-run") as run:
            ...     run.log_metrics({...})
        """
        return ExperimentRun(
            name=name,
            experiments_dir=self.experiments_dir,
        )

    def list_runs(self) -> list[dict[str, Any]]:
        """List semua experiment runs, sorted by start_time descending."""
        dir_path = Path(self.experiments_dir)

        if not dir_path.exists():
            return []

        runs = []
        for filepath in dir_path.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by start_time descending (terbaru di atas)
        runs.sort(key=lambda r: r.get("start_time", ""), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Ambil satu run berdasarkan run_id."""
        for run in self.list_runs():
            if run.get("run_id") == run_id:
                return run
        return None

    def compare_runs(
        self,
        run_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare beberapa runs side-by-side.

        Jika ``run_ids`` tidak diberikan, compare semua completed runs.
        """
        all_runs = self.list_runs()

        if run_ids:
            return [r for r in all_runs if r["run_id"] in run_ids]

        return [r for r in all_runs if r.get("status") == "completed"]

    def get_best_run(self, metric: str = "accuracy") -> dict[str, Any] | None:
        """Ambil run dengan metrik terbaik."""
        completed = [
            r for r in self.list_runs()
            if r.get("status") == "completed"
            and metric in r.get("metrics", {})
        ]

        if not completed:
            return None

        return max(completed, key=lambda r: r["metrics"][metric])

    def delete_run(self, run_id: str) -> bool:
        """Hapus experiment run."""
        dir_path = Path(self.experiments_dir)

        for filepath in dir_path.glob("*.json"):
            if filepath.stem.startswith(run_id):
                filepath.unlink()
                return True

        return False

    def clear_all(self) -> int:
        """Hapus semua experiment runs. Return jumlah yang dihapus."""
        dir_path = Path(self.experiments_dir)
        count = 0

        if dir_path.exists():
            for filepath in dir_path.glob("*.json"):
                filepath.unlink()
                count += 1

        return count

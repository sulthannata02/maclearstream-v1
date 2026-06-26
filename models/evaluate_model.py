"""
CLI: Evaluate semua model yang sudah di-train.

Penggunaan:
    python -m models.evaluate_model
"""

from __future__ import annotations

import sys
from pathlib import Path

# Tambahkan root project ke sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_config, get_paths_config
from services.preprocessing import preprocess
from services.evaluation import evaluate_model, compare_models
from services.prediction import get_available_models
from utils.file_handler import load_artifact
from utils.helpers import format_percentage
from utils.logger import get_logger

logger = get_logger("evaluate_model")


def main() -> None:
    """Evaluate semua model artifacts dan bandingkan."""
    config = get_config()
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    task_type = config.get("project", {}).get("task_type", "classification")

    logger.info("=" * 50)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 50)

    # Cek model yang tersedia
    available = get_available_models()
    if not available:
        logger.error(
            "Belum ada model yang di-train. "
            "Jalankan: python -m models.train_model"
        )
        return

    logger.info(f"Model tersedia: {available}")

    # Preprocessing
    logger.info("Preprocessing dataset...")
    data = preprocess(config)

    # Evaluate setiap model
    all_results: dict[str, dict] = {}

    for model_name in available:
        model_path = artifacts_dir / f"{model_name}.pkl"
        model = load_artifact(model_path)

        metrics = evaluate_model(
            model, data["X_test"], data["y_test"], task_type,
        )
        all_results[model_name] = metrics

        logger.info(f"\n{'='*40}")
        logger.info(f"  {model_name}")
        logger.info(f"{'='*40}")

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                logger.info(f"  {k}: {v:.4f}")

    # Comparison table
    comparison = compare_models(all_results, task_type)
    logger.info(f"\n{'='*50}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*50}")
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Best model
    if task_type == "regression":
        best_metric = "r2"
    else:
        best_metric = "accuracy"

    best_name = max(
        all_results,
        key=lambda n: all_results[n].get(best_metric, 0) or 0,
    )
    best_score = all_results[best_name].get(best_metric, 0)

    logger.info(f"\n{'='*50}")
    logger.info(
        f"🏆 Best Model: {best_name} "
        f"({best_metric}={best_score:.4f})"
    )
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()

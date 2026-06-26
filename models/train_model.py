"""
CLI: Train semua model dari config.

Penggunaan:
    python -m models.train_model
"""

from __future__ import annotations

import sys
from pathlib import Path

# Tambahkan root project ke sys.path agar import bekerja
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_config, get_paths_config
from core.experiment import ExperimentTracker
from services.preprocessing import preprocess
from services.evaluation import evaluate_model
from services.training import train_all_models
from utils.file_handler import save_artifact
from utils.logger import get_logger

logger = get_logger("train_model")


def main() -> None:
    """Train semua model, simpan artifacts, dan log experiment."""
    config = get_config()
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    task_type = config.get("project", {}).get("task_type", "classification")

    logger.info("=" * 50)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 50)

    # 1. Preprocessing
    logger.info("Step 1: Preprocessing dataset...")
    data = preprocess(config)
    logger.info(
        f"Dataset split: train={len(data['X_train'])}, "
        f"test={len(data['X_test'])}"
    )

    # 2. Train semua model
    logger.info("Step 2: Training models...")
    trained_models = train_all_models(
        data["X_train"], data["y_train"], config,
    )

    # 3. Simpan artifacts
    logger.info("Step 3: Menyimpan artifacts...")
    for name, model in trained_models.items():
        model_path = artifacts_dir / f"{name}.pkl"
        save_artifact(model.model, model_path)
        logger.info(f"  Saved: {model_path}")

    # Simpan scaler dan feature names
    if data.get("scaler"):
        save_artifact(data["scaler"], artifacts_dir / "scaler.pkl")
    save_artifact(data["feature_names"], artifacts_dir / "feature_names.pkl")

    # 4. Evaluate dan log experiment
    logger.info("Step 4: Evaluasi dan logging...")
    tracker = ExperimentTracker()

    for name, model in trained_models.items():
        metrics = evaluate_model(
            model, data["X_test"], data["y_test"], task_type,
        )

        with tracker.start_run(name) as run:
            run.log_model(name)
            run.log_params(model.get_params())

            # Log only scalar metrics
            scalar_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }
            run.log_metrics(scalar_metrics)

        # Print results
        logger.info(f"\n{'='*40}")
        logger.info(f"  {model.display_name}")
        logger.info(f"{'='*40}")
        for k, v in scalar_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("✅ Training selesai!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

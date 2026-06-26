"""
Pipeline Engine — Chainable preprocessing steps.

Setiap step adalah class dengan method ``transform(df) → df``.
Pipeline memproses steps secara berurutan dan bisa di-build dari config.

Contoh penggunaan:
    >>> pipeline = Pipeline.from_config(config)
    >>> result = pipeline.run(df, target_column="target")
    >>> X_train, X_test = result["X_train"], result["X_test"]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)

from core.exceptions import PipelineError


# ══════════════════════════════════════════════
# Base Step
# ══════════════════════════════════════════════

class BaseStep:
    """Abstract base class untuk pipeline step."""

    name: str = "base_step"
    description: str = "Base step"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform dataframe. Override di subclass."""
        raise NotImplementedError

    def describe(self) -> str:
        """Deskripsi step untuk display di UI."""
        return self.description


# ══════════════════════════════════════════════
# Built-in Steps
# ══════════════════════════════════════════════

class MissingValueHandler(BaseStep):
    """Handle missing values (NaN dan zeros pada kolom tertentu)."""

    name = "missing_values"

    def __init__(
        self,
        strategy: str = "median",
        zero_columns: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.zero_columns = zero_columns or []
        self.description = f"Handle missing values (strategy: {strategy})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        df = df.copy()

        # Ganti 0 → NaN pada kolom yang ditentukan
        if self.zero_columns:
            for col in self.zero_columns:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)

        # Isi NaN berdasarkan strategi
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if self.strategy == "median":
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        elif self.strategy == "mean":
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        elif self.strategy == "mode":
            for col in df.columns:
                if df[col].isna().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
        elif self.strategy == "drop":
            df = df.dropna().reset_index(drop=True)

        return df


class OutlierHandler(BaseStep):
    """Deteksi dan handle outlier."""

    name = "outlier_removal"

    def __init__(
        self,
        method: str = "none",
        threshold: float = 1.5,
    ) -> None:
        self.method = method
        self.threshold = threshold
        self.description = f"Outlier handling (method: {method})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        target = kwargs.get("target_column")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if target and target in numeric_cols:
            numeric_cols = numeric_cols.drop(target)

        mask = pd.Series(True, index=df.index)

        if self.method == "iqr":
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.threshold * iqr
                upper = q3 + self.threshold * iqr
                mask &= (df[col] >= lower) & (df[col] <= upper)

        elif self.method == "zscore":
            for col in numeric_cols:
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask &= z <= self.threshold

        return df[mask].reset_index(drop=True)


class CategoricalEncoder(BaseStep):
    """Encode kolom kategorikal."""

    name = "encoding"

    def __init__(self, method: str = "label") -> None:
        self.method = method
        self.encoders: dict[str, LabelEncoder] = {}
        self.description = f"Categorical encoding (method: {method})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols) == 0:
            return df

        if self.method == "label":
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le

        elif self.method == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df


class FeatureScaler(BaseStep):
    """Scaling fitur numerik."""

    name = "scaling"

    SCALERS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(self, method: str = "standard") -> None:
        self.method = method
        self.scaler = None
        self.description = f"Feature scaling (method: {method})"

    def get_scaler(self):
        """Return fitted scaler instance (untuk prediction)."""
        return self.scaler

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        target = kwargs.get("target_column")

        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in feature_cols:
            feature_cols.remove(target)

        if not feature_cols:
            return df

        scaler_class = self.SCALERS.get(self.method)
        if scaler_class is None:
            raise PipelineError(
                self.name,
                f"Scaler '{self.method}' tidak dikenal. "
                f"Pilihan: {list(self.SCALERS.keys())}",
            )

        self.scaler = scaler_class()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df


class DataSplitter(BaseStep):
    """Split dataset menjadi train dan test set."""

    name = "split"

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.description = (
            f"Data split (test_size: {test_size}, stratify: {stratify})"
        )

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        # DataSplitter tidak mengubah DataFrame secara langsung.
        # Hasil split diakses via Pipeline.run().
        return df

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> tuple[Any, Any, Any, Any]:
        """Eksekusi split dan return (X_train, X_test, y_train, y_test)."""
        stratify_param = y if self.stratify else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )


# ══════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════

# Mapping nama step → class
STEP_REGISTRY: dict[str, type[BaseStep]] = {
    "missing_values": MissingValueHandler,
    "outlier_removal": OutlierHandler,
    "encoding": CategoricalEncoder,
    "scaling": FeatureScaler,
    "split": DataSplitter,
}


class Pipeline:
    """
    Pipeline preprocessing yang chainable.

    Parameters
    ----------
    steps : list[BaseStep]
        Urutan step preprocessing.
    """

    def __init__(self, steps: list[BaseStep] | None = None) -> None:
        self.steps: list[BaseStep] = steps or []
        self._scaler_step: FeatureScaler | None = None
        self._splitter_step: DataSplitter | None = None

    def add_step(self, step: BaseStep) -> "Pipeline":
        """Tambahkan step ke pipeline."""
        self.steps.append(step)
        return self

    def describe(self) -> list[dict[str, str]]:
        """Kembalikan deskripsi semua steps untuk UI display."""
        return [
            {"name": step.name, "description": step.describe()}
            for step in self.steps
        ]

    def run(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        """
        Jalankan semua pipeline steps dan return hasil split.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe mentah.
        target_column : str
            Nama kolom target.

        Returns
        -------
        dict
            Berisi: dataframe, feature_names, scaler, X_train, X_test,
            y_train, y_test.
        """
        processed_df = df.copy()

        for step in self.steps:
            if isinstance(step, DataSplitter):
                self._splitter_step = step
                continue

            if isinstance(step, FeatureScaler):
                self._scaler_step = step

            try:
                processed_df = step.transform(
                    processed_df,
                    target_column=target_column,
                )
            except Exception as e:
                raise PipelineError(step.name, str(e)) from e

        # Pisahkan fitur dan target
        if target_column not in processed_df.columns:
            raise PipelineError(
                "split",
                f"Kolom target '{target_column}' tidak ditemukan setelah "
                "preprocessing.",
            )

        X = processed_df.drop(columns=[target_column])
        y = processed_df[target_column]
        feature_names = X.columns.tolist()

        # Split
        if self._splitter_step:
            X_train, X_test, y_train, y_test = self._splitter_step.split(X, y)
        else:
            # Default split jika tidak ada DataSplitter step
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )

        # Ambil scaler (untuk prediction nanti)
        scaler = None
        if self._scaler_step:
            scaler = self._scaler_step.get_scaler()

        return {
            "dataframe": processed_df,
            "feature_names": feature_names,
            "scaler": scaler,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """
        Build pipeline dari config dict (section 'pipeline').

        Parameters
        ----------
        config : dict
            Full config dict (seluruh config.yaml).

        Returns
        -------
        Pipeline
            Pipeline instance yang siap dijalankan.
        """
        pipeline = cls()
        pipeline_config = config.get("pipeline", {})
        dataset_config = config.get("dataset", {})
        zero_columns = dataset_config.get("zero_columns", [])

        steps_config = pipeline_config.get("steps", [])

        for step_cfg in steps_config:
            step_name = step_cfg.get("name", "")
            step_class = STEP_REGISTRY.get(step_name)

            if step_class is None:
                raise PipelineError(
                    step_name,
                    f"Step '{step_name}' tidak dikenal. "
                    f"Pilihan: {list(STEP_REGISTRY.keys())}",
                )

            # Build kwargs dari config (hilangkan key 'name')
            kwargs = {k: v for k, v in step_cfg.items() if k != "name"}

            # Inject zero_columns untuk MissingValueHandler
            if step_name == "missing_values":
                kwargs["zero_columns"] = zero_columns

            try:
                step = step_class(**kwargs)
            except TypeError as e:
                raise PipelineError(
                    step_name,
                    f"Parameter tidak valid: {e}",
                ) from e

            pipeline.add_step(step)

        return pipeline

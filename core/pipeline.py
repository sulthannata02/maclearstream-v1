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

import os
import hashlib
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore # pyright: ignore[reportMissingImports]
import pandas as pd  # type: ignore # pyright: ignore[reportMissingImports]
import joblib  # type: ignore # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import (  # type: ignore # pyright: ignore[reportMissingImports]
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore # pyright: ignore[reportMissingImports]

from core.exceptions import PipelineError  # type: ignore # pyright: ignore[reportMissingImports]


# ══════════════════════════════════════════════
# Base Step
# ══════════════════════════════════════════════

class BaseStep:
    """Abstract base class untuk pipeline step."""

    name: str = "base_step"
    description: str = "Base step"
    optional: bool = False

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
        **kwargs: Any,
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
        **kwargs: Any,
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

    def __init__(self, method: str = "label", **kwargs: Any) -> None:
        self.method = method
        self.encoders: dict[str, LabelEncoder] = {}
        self.description = f"Categorical encoding (method: {method})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        target = kwargs.get("target_column")
        if target and target in cat_cols:
            cat_cols = cat_cols.drop(target)

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

    def __init__(self, method: str = "standard", **kwargs: Any) -> None:
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


class NLPCleaner(BaseStep):
    """Membersihkan teks untuk NLP workflows."""

    name = "nlp_cleaning"

    def __init__(
        self,
        enabled: bool = False,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        **kwargs: Any,
    ) -> None:
        self.enabled = enabled
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.description = f"NLP Text Cleaning (enabled: {enabled}, lowercase: {lowercase})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not self.enabled:
            return df

        df = df.copy()
        text_cols = df.select_dtypes(include=["object"]).columns
        target = kwargs.get("target_column")
        if target and target in text_cols:
            text_cols = text_cols.drop(target)

        for col in text_cols:
            if self.lowercase:
                df[col] = df[col].astype(str).str.lower()
            if self.remove_stopwords:
                df[col] = df[col].astype(str).str.replace(r'\b(the|is|in|at|of|on|and|a|an)\b', '', regex=True)
                df[col] = df[col].str.strip()

        return df


class TextVectorizer(BaseStep):
    """Mengubah teks menjadi representasi numerik (TF-IDF atau BoW)."""

    name = "vectorizer"

    def __init__(
        self,
        method: str = "none",
        max_features: int = 5000,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.max_features = max_features
        self.vectorizers: dict[str, Any] = {}
        self.description = f"Text Vectorization (method: {method}, max_features: {max_features})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.method == "none":
            return df

        df = df.copy()
        text_cols = df.select_dtypes(include=["object"]).columns
        target = kwargs.get("target_column")
        if target and target in text_cols:
            text_cols = text_cols.drop(target)

        if len(text_cols) == 0:
            return df

        for col in text_cols:
            vec_class = TfidfVectorizer if self.method == "tfidf" else CountVectorizer
            vec = vec_class(max_features=self.max_features)
            transformed = vec.fit_transform(df[col].astype(str)).toarray()

            col_names = [f"{col}_{self.method}_{i}" for i in range(transformed.shape[1])]
            vec_df = pd.DataFrame(transformed, columns=col_names, index=df.index)

            df = pd.concat([df.drop(columns=[col]), vec_df], axis=1)
            self.vectorizers[col] = vec

        return df


class CustomTransformer(BaseStep):
    """Custom user-defined transformation."""

    name = "custom_transform"

    def __init__(self, func_name: str = "none", **kwargs: Any) -> None:
        self.func_name = func_name
        self.description = f"Custom transformation ({func_name})"

    def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if self.func_name == "none":
            return df
        df = df.copy()
        if self.func_name == "log_transform":
            num_cols = df.select_dtypes(include=[np.number]).columns
            target = kwargs.get("target_column")
            if target and target in num_cols:
                num_cols = num_cols.drop(target)
            for col in num_cols:
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
        return df


class DataSplitter(BaseStep):
    """Split dataset menjadi train dan test set."""

    name = "split"

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        **kwargs: Any,
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
    "nlp_cleaning": NLPCleaner,
    "vectorizer": TextVectorizer,
    "custom_transform": CustomTransformer,
    "split": DataSplitter,
}


class Pipeline:
    """
    Pipeline preprocessing yang chainable.

    Parameters
    ----------
    steps : list[BaseStep]
        Urutan step preprocessing.
    cache_enabled : bool
        Mengaktifkan caching via joblib.
    """

    def __init__(
        self,
        steps: list[BaseStep] | None = None,
        cache_enabled: bool = False,
    ) -> None:
        self.steps: list[BaseStep] = steps or []
        self.cache_enabled: bool = cache_enabled
        self._scaler_step: FeatureScaler | None = None
        self._splitter_step: DataSplitter | None = None
        self._encoders: dict[str, Any] = {}

        if self.cache_enabled:
            cache_dir = Path("artifacts/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.memory = joblib.Memory(location=str(cache_dir), verbose=0)
        else:
            self.memory = joblib.Memory(location=None, verbose=0)

    def add_step(self, step: BaseStep) -> "Pipeline":
        """Tambahkan step ke pipeline."""
        self.steps.append(step)
        return self

    def describe(self) -> list[dict[str, str]]:
        """Kembalikan deskripsi semua steps untuk UI display."""
        return [
            {"name": step.name, "description": step.describe(), "optional": str(getattr(step, "optional", False))}
            for step in self.steps
        ]

    def _execute_steps(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, DataSplitter | None, FeatureScaler | None, dict[str, Any]]:
        processed_df = df.copy()
        splitter = None
        scaler = None
        encoders = {}

        for step in self.steps:
            if isinstance(step, DataSplitter):
                splitter = step
                continue

            if isinstance(step, FeatureScaler):
                scaler = step

            if isinstance(step, CategoricalEncoder):
                encoders.update(getattr(step, "encoders", {}))

            try:
                processed_df = step.transform(
                    processed_df,
                    target_column=target_column,
                )
            except Exception as e:
                raise PipelineError(step.name, str(e)) from e

        return processed_df, splitter, scaler, encoders

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
            Berisi: dataframe, feature_names, scaler, encoders, X_train, X_test,
            y_train, y_test.
        """
        if self.cache_enabled:
            cached_exec = self.memory.cache(self._execute_steps)
            processed_df, self._splitter_step, self._scaler_step, self._encoders = cached_exec(df, target_column)
        else:
            processed_df, self._splitter_step, self._scaler_step, self._encoders = self._execute_steps(df, target_column)

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
            "encoders": self._encoders,
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
        pipeline_config = config.get("pipeline", {})
        cache_enabled = pipeline_config.get("cache_enabled", False)
        pipeline = cls(cache_enabled=cache_enabled)

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

            # Build kwargs dari config (hilangkan key 'name' dan 'optional')
            kwargs = {k: v for k, v in step_cfg.items() if k not in ("name", "optional")}
            optional_flag = step_cfg.get("optional", False)

            # Inject zero_columns untuk MissingValueHandler
            if step_name == "missing_values":
                kwargs["zero_columns"] = zero_columns

            try:
                step = step_class(**kwargs)
                step.optional = optional_flag
            except TypeError as e:
                raise PipelineError(
                    step_name,
                    f"Parameter tidak valid: {e}",
                ) from e

            pipeline.add_step(step)

        return pipeline


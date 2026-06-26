"""
Custom exceptions untuk framework ML × Streamlit.

Setiap exception punya pesan user-friendly yang bisa langsung
ditampilkan di UI Streamlit via ``st.error()``.
"""


class MLFrameworkError(Exception):
    """Base exception untuk semua error framework."""

    def __init__(self, message: str, hint: str = "") -> None:
        self.hint = hint
        super().__init__(message)


class ModelNotTrainedError(MLFrameworkError):
    """Dilempar ketika model belum di-train tapi dicoba digunakan."""

    def __init__(self, model_name: str = "") -> None:
        msg = "Model belum di-train."
        if model_name:
            msg = f"Model '{model_name}' belum di-train."
        hint = (
            "Jalankan training terlebih dahulu melalui:\n"
            "• UI: halaman Training → klik Train\n"
            "• CLI: python -m models.train_model"
        )
        super().__init__(msg, hint=hint)


class DatasetNotFoundError(MLFrameworkError):
    """Dilempar ketika file dataset tidak ditemukan."""

    def __init__(self, path: str = "") -> None:
        msg = "Dataset tidak ditemukan."
        if path:
            msg = f"Dataset tidak ditemukan: {path}"
        hint = (
            "Pastikan file dataset sudah ada di folder dataset/.\n"
            "Kemudian sesuaikan path di config.yaml → dataset.path"
        )
        super().__init__(msg, hint=hint)


class ConfigError(MLFrameworkError):
    """Dilempar ketika konfigurasi tidak valid."""

    def __init__(self, message: str = "Konfigurasi tidak valid.") -> None:
        hint = "Periksa file config.yaml dan pastikan format YAML benar."
        super().__init__(message, hint=hint)


class PipelineError(MLFrameworkError):
    """Dilempar ketika pipeline preprocessing gagal."""

    def __init__(self, step_name: str = "", detail: str = "") -> None:
        msg = "Pipeline preprocessing gagal."
        if step_name:
            msg = f"Pipeline gagal di step '{step_name}'."
        if detail:
            msg += f" Detail: {detail}"
        hint = "Periksa konfigurasi pipeline di config.yaml → pipeline.steps"
        super().__init__(msg, hint=hint)


class ValidationError(MLFrameworkError):
    """Dilempar ketika validasi data gagal."""

    def __init__(self, message: str = "Validasi data gagal.") -> None:
        hint = "Periksa dataset dan pastikan format sesuai dengan config."
        super().__init__(message, hint=hint)

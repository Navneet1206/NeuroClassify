import os
from typing import Optional

try:
    # Optional: load from api_app/.env if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Environment-driven settings
MODEL_API_URL: Optional[str] = os.getenv("MODEL_API_URL")  # e.g., https://your-inference-provider/api/v1/predict
MODEL_API_KEY: Optional[str] = os.getenv("MODEL_API_KEY")  # Bearer or custom header
TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))

# App
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")

# Classes (must match your external model’s classes/order)
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "normal"]

# OpenRouter (optional) — if provided, backend will call OpenRouter to classify images via LLM
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_APP_TITLE: str = os.getenv("OPENROUTER_APP_TITLE", "NeuroClassify-API")
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
OPENROUTER_MAX_TOKENS: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "256"))

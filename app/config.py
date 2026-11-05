"""Runtime configuration. Every limit and secret comes from environment variables
(or a local, git-ignored .env file). Nothing secret is hard-coded here."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')

    host: str = '127.0.0.1'
    port: int = 8000
    log_level: str = 'INFO'

    # --- Model / provider -------------------------------------------------
    model_provider: str = 'unconfigured'            # 'gemini' or 'unconfigured'
    model_name: str = ''                            # default model
    available_models: str = ''                      # comma-separated list for the UI
    gemini_api_key: str = ''

    # --- Execution budget (the system, not the model, owns these) ---------
    max_steps: int = Field(default=6, ge=1, le=6)               # model calls per run
    max_tool_retries: int = Field(default=2, ge=0, le=2)        # extra attempts per tool call
    max_repair_attempts: int = Field(default=2, ge=0, le=2)     # contract repairs per run
    max_output_tokens: int = Field(default=768, ge=64, le=4096)
    max_run_tokens: int = Field(default=40000, ge=1000)         # input+output tokens per run
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    llm_timeout_seconds: float = Field(default=15, gt=0, le=40)
    tool_timeout_seconds: float = Field(default=2.5, gt=0, le=10)

    # --- Conversation memory ----------------------------------------------
    max_history_messages: int = Field(default=12, ge=2, le=40)  # 6 user/assistant turns
    max_history_chars: int = Field(default=12000, ge=1000)
    max_sessions: int = Field(default=100, ge=1)

    # --- Approximate prices, USD per 1M tokens: "model=in/out;model=in/out"
    model_prices: str = ('gemini-3.1-flash-lite=0.25/1.50;gemini-3.5-flash-lite=0.30/2.50;'
                         'gemini-2.5-flash=0.30/2.50')
    # If the active model errors (rate limit, outage), switch once to the next model in
    # MODEL_NAME + AVAILABLE_MODELS order.
    model_fallback: bool = True

    def models(self) -> list[str]:
        names = [self.model_name] if self.model_name else []
        names += [m.strip() for m in self.available_models.split(',') if m.strip()]
        return list(dict.fromkeys(names))  # de-duplicate, keep order

    def price(self, model: str) -> tuple[float, float] | None:
        for item in self.model_prices.split(';'):
            name, _, rates = item.partition('=')
            if name.strip() == model and '/' in rates:
                inp, out = rates.split('/', 1)
                try:
                    return float(inp), float(out)
                except ValueError:
                    return None
        return None


settings = Settings()

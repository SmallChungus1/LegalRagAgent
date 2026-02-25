"""LLM configuration — single entry point for all LLM calls.

Supports multiple providers via LLM_PROVIDER env var:
  gemma, gemini-flash, gemini-flash-lite,
  groq-llama70b, groq-llama8b, groq-maverick, groq-scout, groq-gpt120b, groq-kimi, groq-qwen,
  or-llama70b, or-gpt20b, or-gemma27b, or-qwen3-coder, or-nemotron, or-mistral, or-hermes,
  ollama, cerebras

Falls back to raw LLM_BASE_URL/LLM_API_KEY/LLM_MODEL if LLM_PROVIDER is not set.
"""

import functools
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Provider registry: name -> (base_url, api_key_env, model, rpd, tpd)
PROVIDERS = {
    # --- DeepSeek ---
    "deepseek":           ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat",            None, None),
    "deepseek-reasoner":  ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-reasoner",        None, None),
    # --- Google AI Studio ---
    "gemma":              ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemma-3-27b-it",       14_400, None),
    "gemini-flash":       ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.5-flash",     20,     None),
    "gemini-flash-lite":  ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.5-flash-lite", 20,    None),
    # --- Groq ---
    "groq-llama70b":      ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile",                    1_000, 100_000),
    "groq-llama8b":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.1-8b-instant",                      14_400, 500_000),
    "groq-maverick":      ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "meta-llama/llama-4-maverick-17b-128e-instruct", 1_000, 500_000),
    "groq-scout":         ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "meta-llama/llama-4-scout-17b-16e-instruct", 1_000, 500_000),
    "groq-gpt120b":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b",                       1_000, 200_000),
    "groq-kimi":          ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "moonshotai/kimi-k2-instruct",               1_000, 300_000),
    "groq-qwen":          ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3-32b",                            1_000, 500_000),
    # --- OpenRouter (free tier — weekly token limits, no RPD cap) ---
    "or-llama70b":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free",     None, None),
    "or-gpt20b":          ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-oss-20b:free",                    None, None),
    "or-gemma27b":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "google/gemma-3-27b-it:free",                 None, None),
    "or-qwen3-coder":     ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "qwen/qwen3-coder-480b-a35b:free",           None, None),
    "or-nemotron":        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "nvidia/nemotron-nano-9b-v2:free",            None, None),
    "or-mistral":         ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "mistralai/mistral-small-3.1-24b-instruct:free", None, None),
    "or-hermes":          ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "nousresearch/hermes-3-llama-3.1-405b:free",  None, None),
    # --- Cerebras ---
    "cerebras":           ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "llama-3.3-70b",                              14_000, 1_000_000),
    # --- Ollama (local) ---
    "ollama":             ("http://localhost:11434/v1", None, "llama3",                                                     None,  None),
}


def _resolve_provider():
    """Resolve (base_url, api_key, model) from LLM_PROVIDER or legacy env vars."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider and provider in PROVIDERS:
        base_url, key_env, model, _, _ = PROVIDERS[provider]
        api_key = os.getenv(key_env, "") if key_env else "ollama"
        return base_url, api_key, model

    # Legacy fallback: raw env vars
    return (
        os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1"),
        os.getenv("LLM_API_KEY", "no-key-set"),
        os.getenv("LLM_MODEL", "llama-3.3-70b"),
    )


def get_provider_info() -> dict:
    """Return current provider name, model, and rate limits (for eval logging)."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider and provider in PROVIDERS:
        base_url, key_env, model, rpd, tpd = PROVIDERS[provider]
        return {"provider": provider, "model": model, "rpd": rpd, "tpd": tpd}
    return {
        "provider": "custom",
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b"),
        "rpd": None,
        "tpd": None,
    }


@functools.lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Returns a cached ChatOpenAI instance configured from environment variables."""
    base_url, api_key, model = _resolve_provider()
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


def list_providers():
    """Print all available providers with their rate limits."""
    print(f"\n{'Provider':<22} {'Model':<52} {'RPD':>8} {'TPD':>10}")
    print("-" * 95)
    for name, (_, _, model, rpd, tpd) in sorted(PROVIDERS.items()):
        rpd_str = f"{rpd:,}" if rpd else "local"
        tpd_str = f"{tpd:,}" if tpd else "local"
        print(f"{name:<22} {model:<52} {rpd_str:>8} {tpd_str:>10}")
    print()
    info = get_provider_info()
    print(f"Active: {info['provider']} ({info['model']})")


if __name__ == "__main__":
    list_providers()

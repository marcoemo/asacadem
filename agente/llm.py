"""Cliente LLM vía OpenRouter (API compatible con OpenAI)."""
from langchain_openai import ChatOpenAI

from . import config


def get_llm() -> ChatOpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "Falta OPENROUTER_API_KEY. Copia .env.example a .env y pega tu API key."
        )
    return ChatOpenAI(
        model=config.MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=0,
        # esfuerzo de razonamiento bajo = menos costo (recomendación del docente)
        extra_body={"reasoning": {"effort": config.REASONING_EFFORT}},
    )

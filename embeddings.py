"""Embeddings con dos backends intercambiables.

- openrouter: embeddings semánticos reales (API compatible con OpenAI). Costo mínimo.
- local: bolsa de palabras con hashing (sin red, sin costo). Léxico, no semántico;
  se usa como respaldo para que el sistema siempre funcione.
En modo "auto" se intenta openrouter y, si falla, se cae a local con un aviso.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from . import config


_STOP = set("de la el los las un una unos unas y o a en por para con sin que se su sus al del lo es son "
            "mi me mis tu te yo qué que cómo como cuál cual cuando cuándo donde dónde si no ya hay "
            "puedo debo necesito tengo quiero más mas muy este esta esto ese esa hasta cuántos cuantos cuántas "
            "cuantas cuánto cuanto tiene tienen hacer".split())


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


_STOP_NORM = {_normalizar(w) for w in _STOP}


def _tokens(texto: str) -> list[str]:
    """Palabras normalizadas, sin stopwords, recortadas a 5 letras (stemming barato)."""
    palabras = re.findall(r"[a-z0-9]+", _normalizar(texto))
    palabras = [p for p in palabras if p not in _STOP_NORM and len(p) > 1]
    return [p[:5] for p in palabras]


class LocalEmbedder:
    """Bolsa de palabras (unigramas+bigramas con stemming) vía hashing. Sin red ni costo."""

    name = "local-hash-v2"
    min_score = 0.2  # umbral de relevancia (calibrado con los documentos de ejemplo)
    min_score_memoria = 0.1  # los recuerdos son textos cortos: puntajes más bajos

    def __init__(self) -> None:
        self._vec = HashingVectorizer(
            n_features=2**15, analyzer=lambda t: (lambda w: w + [a + "_" + b for a, b in zip(w, w[1:])])(_tokens(t)),
            alternate_sign=False, norm="l2",
        )

    def embed(self, textos: list[str]) -> np.ndarray:
        return self._vec.transform(textos).toarray().astype(np.float32)


class OpenRouterEmbedder:
    min_score = 0.25
    min_score_memoria = 0.25

    def __init__(self) -> None:
        from openai import OpenAI

        self.name = f"openrouter:{config.EMBEDDING_MODEL}"
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)

    def embed(self, textos: list[str]) -> np.ndarray:
        salida: list[list[float]] = []
        for i in range(0, len(textos), 64):
            r = self._client.embeddings.create(model=config.EMBEDDING_MODEL, input=textos[i : i + 64])
            salida.extend(d.embedding for d in r.data)
        m = np.array(salida, dtype=np.float32)
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


_cache = None


def get_embedder():
    global _cache
    if _cache is not None:
        return _cache
    backend = config.EMBEDDING_BACKEND
    if backend in ("auto", "openrouter") and config.OPENROUTER_API_KEY:
        try:
            emb = OpenRouterEmbedder()
            emb.embed(["ping"])  # prueba de conectividad
            _cache = emb
            return _cache
        except Exception as e:  # noqa: BLE001
            if backend == "openrouter":
                raise
            print(f"[aviso] Embeddings por OpenRouter no disponibles ({type(e).__name__}); "
                  "uso embeddings locales.")
    _cache = LocalEmbedder()
    return _cache


def reset_embedder() -> None:
    global _cache
    _cache = None

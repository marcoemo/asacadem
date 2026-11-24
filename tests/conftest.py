"""Configuración de tests: almacenamiento temporal, embeddings locales y un LLM falso guionado."""
import os
import tempfile

os.environ["AGENTE_STORAGE_DIR"] = tempfile.mkdtemp(prefix="agente_test_")
os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["OPENROUTER_API_KEY"] = ""

import pytest
from helpers_agente import LLMGuionado, ia  # noqa: F401


@pytest.fixture
def llm_guionado():
    def _crear(*respuestas):
        return LLMGuionado(respuestas=list(respuestas), llamadas=[])
    return _crear


@pytest.fixture(autouse=True)
def limpiar_servicios():
    from agente import servicios
    servicios.reset_servicios()
    yield

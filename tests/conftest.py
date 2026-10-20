"""Configuración de tests: almacenamiento temporal, embeddings locales y un LLM falso guionado."""
import os
import tempfile

os.environ["AGENTE_STORAGE_DIR"] = tempfile.mkdtemp(prefix="agente_test_")
os.environ["EMBEDDING_BACKEND"] = "local"
os.environ["OPENROUTER_API_KEY"] = ""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMGuionado(BaseChatModel):
    """Devuelve respuestas predefinidas en orden. Permite probar el grafo sin red ni costo."""

    respuestas: list = []
    llamadas: list = []

    @property
    def _llm_type(self) -> str:
        return "guionado"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.llamadas.append(messages)
        msg = self.respuestas.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def ia(texto="", tool=None, args=None, id_="call_1"):
    if tool:
        return AIMessage(content=texto, tool_calls=[{"name": tool, "args": args or {}, "id": id_}])
    return AIMessage(content=texto)


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

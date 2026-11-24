"""Utilidades de prueba: LLM falso guionado (sin red ni costo).

El nombre del módulo es único a propósito (evita choques con otros paquetes llamados `tests`).
"""
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

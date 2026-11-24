"""Observabilidad: mide cada turno del agente y lo registra como un evento JSONL.

Por turno se registran: latencia total, por nodo del grafo, por llamada al modelo y por tool; tokens;
número de llamadas al LLM; tools usadas y sus errores; ruta elegida; memoria del proceso; y, si se
conoce el resultado esperado, si el agente decidió correctamente. Los textos se guardan SIN datos
personales (se redactan antes de escribir el log).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from . import config
from .seguridad import redactar_pii

NODOS = {"guardia", "responder_bloqueo", "recuperar_contexto", "planificar", "agente", "tools",
         "responder_directo", "resumir"}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _memoria_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:  # noqa: BLE001
        return 0.0


class ObservabilidadCallback(BaseCallbackHandler):
    """Recoge tiempos y tokens de nodos, llamadas al modelo y tools durante un turno."""

    def __init__(self) -> None:
        self.nodos: dict[str, float] = defaultdict(float)
        self.llm_ms: list[float] = []
        self.tokens_in = self.tokens_out = 0
        self.tools: list[dict] = []
        self._ini: dict[Any, tuple[str, float, str]] = {}

    # --- nodos del grafo
    def on_chain_start(self, serialized, inputs, *, run_id, name=None, metadata=None, **kw):
        nodo = (metadata or {}).get("langgraph_node") or name
        if nodo in NODOS and name == nodo:
            self._ini[run_id] = ("nodo", time.perf_counter(), nodo)

    def on_chain_end(self, outputs, *, run_id, **kw):
        ini = self._ini.pop(run_id, None)
        if ini and ini[0] == "nodo":
            self.nodos[ini[2]] += _ms(ini[1])

    # --- modelo
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._ini[run_id] = ("llm", time.perf_counter(), "llm")

    def on_llm_start(self, serialized, prompts, *, run_id, **kw):
        self._ini[run_id] = ("llm", time.perf_counter(), "llm")

    def on_llm_end(self, response, *, run_id, **kw):
        ini = self._ini.pop(run_id, None)
        if ini:
            self.llm_ms.append(_ms(ini[1]))
        try:
            um = response.generations[0][0].message.usage_metadata or {}
            self.tokens_in += int(um.get("input_tokens", 0))
            self.tokens_out += int(um.get("output_tokens", 0))
        except Exception:  # noqa: BLE001
            pass

    def on_llm_error(self, error, *, run_id, **kw):
        ini = self._ini.pop(run_id, None)
        if ini:
            self.llm_ms.append(_ms(ini[1]))

    # --- tools
    def on_tool_start(self, serialized, input_str, *, run_id, name=None, **kw):
        nombre = name or (serialized or {}).get("name", "tool")
        self._ini[run_id] = ("tool", time.perf_counter(), nombre)

    def on_tool_end(self, output, *, run_id, **kw):
        ini = self._ini.pop(run_id, None)
        if not ini:
            return
        contenido = output.content if hasattr(output, "content") else str(output)
        estado = ("error" if str(contenido).startswith("ERROR") else
                  "sin_resultados" if "SIN_RESULTADOS" in str(contenido) else "ok")
        self.tools.append({"nombre": ini[2], "ms": _ms(ini[1]), "estado": estado})

    def on_tool_error(self, error, *, run_id, **kw):
        ini = self._ini.pop(run_id, None)
        if ini:
            self.tools.append({"nombre": ini[2], "ms": _ms(ini[1]), "estado": "excepcion"})


def registrar(evento: dict, path=None) -> None:
    path = path or config.LOG_PATH
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")


def evaluar_esperado(evento: dict, esperado: dict) -> dict:
    """Compara lo que hizo el agente con lo esperado: ruta, tools requeridas y tools prohibidas."""
    usadas = {t["nombre"] for t in evento["tools"]}
    ok_ruta = evento["ruta"] == esperado.get("ruta", evento["ruta"])
    ok_tools = (set(esperado.get("tools_requeridas", [])) <= usadas
                and not (set(esperado.get("tools_prohibidas", [])) & usadas))
    return {"correcto_ruta": ok_ruta, "correcto_tools": ok_tools,
            "correcto": bool(ok_ruta and ok_tools and not evento.get("excepcion"))}


def medir_turno(grafo, texto: str, user_id: str, thread_id: str, *, escenario: str = "manual",
                consulta_id: str | None = None, repeticion: int = 0, esperado: dict | None = None,
                registrar_log: bool = True) -> dict:
    """Ejecuta un turno midiendo todo y devuelve (y registra) el evento."""
    cb = ObservabilidadCallback()
    cfg = {"configurable": {"user_id": user_id, "thread_id": thread_id}, "recursion_limit": 20,
           "callbacks": [cb]}
    t0 = time.perf_counter()
    excepcion, res = None, {}
    try:
        res = grafo.invoke({"messages": [HumanMessage(content=texto)]}, cfg)
    except Exception as e:  # noqa: BLE001
        excepcion = f"{type(e).__name__}: {e}"[:300]
    total = _ms(t0)

    msgs = res.get("messages", [])
    ult_h = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=-1)
    nuevos = msgs[ult_h + 1:]
    respuesta = next((m.content for m in reversed(nuevos) if isinstance(m, AIMessage) and not m.tool_calls), "")
    respuesta = respuesta if isinstance(respuesta, str) else str(respuesta)
    plan = res.get("plan", {}) or {}
    ruta = ("bloqueado" if res.get("bloqueado") else
            "saludo" if plan.get("intencion") == "saludo" else "agente")
    entrada, pii_in = redactar_pii(texto)
    salida, _ = redactar_pii(respuesta)

    evento = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "escenario": escenario, "consulta_id": consulta_id,
        "repeticion": repeticion, "usuario": user_id, "entrada": entrada, "salida": salida, "ruta": ruta,
        "intencion": plan.get("intencion"), "urgencia": plan.get("urgencia"),
        "plan_fallback": bool(plan.get("fallback")), "alerta": res.get("alerta", ""),
        "pii": res.get("pii", pii_in), "tools": cb.tools,
        "errores_tool": sum(t["estado"] in ("error", "excepcion") for t in cb.tools),
        "sin_resultados": sum(t["estado"] == "sin_resultados" for t in cb.tools),
        "excepcion": excepcion, "llamadas_llm": len(cb.llm_ms), "latencia_llm_ms": round(sum(cb.llm_ms), 2),
        "tokens_in": cb.tokens_in, "tokens_out": cb.tokens_out,
        "costo_usd": round((cb.tokens_in * config.PRECIO_IN_USD_1M + cb.tokens_out * config.PRECIO_OUT_USD_1M) / 1e6, 6),
        "latencia_ms": total, "latencia_nodos": {k: round(v, 2) for k, v in cb.nodos.items()},
        "memoria_mb": _memoria_mb(), "cita_fuente": ".md" in respuesta,
    }
    if esperado is not None:
        evento.update(evaluar_esperado(evento, esperado))
    if registrar_log:
        registrar(evento)
    return evento

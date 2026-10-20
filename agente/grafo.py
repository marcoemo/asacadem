"""Orquestación del agente con LangGraph.

START -> recuperar_contexto -> planificar -+-> responder_directo -+
                                           |                      |
                                           +-> agente <-> tools   +-> resumir -> END

* recuperar_contexto : memoria de largo plazo (búsqueda semántica) -> state.memorias
* planificar         : el LLM clasifica la consulta y arma un plan priorizado (JSON)
* agente / tools     : bucle ReAct (el LLM decide qué tool llamar y cuándo parar)
* responder_directo  : ruta corta sin tools (saludos / charla), decisión adaptativa
* resumir            : gestión de ventana de contexto (resume mensajes antiguos)
La memoria de corto plazo es el checkpointer SQLite (persistencia por thread_id).
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, RemoveMessage,
                                     SystemMessage)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from . import config as cfg
from .servicios import get_memoria
from .tools import TOOLS


class Estado(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    resumen: str          # resumen de la conversación antigua (ventana de contexto)
    memorias: str         # recuerdos de largo plazo relevantes para este turno
    plan: dict            # plan priorizado generado por el planificador


PROMPT_PLANIFICADOR = """Eres el planificador de un agente de soporte académico universitario.
Analiza el último mensaje del estudiante y responde SOLO con un JSON válido, sin texto extra:
{"intencion": "saludo" | "consulta" | "tramite" | "escalamiento",
 "urgencia": "alta" | "media" | "baja",
 "subtareas": [{"orden": 1, "tarea": "...", "prioridad": "alta" | "media" | "baja"}]}
Reglas: "saludo" es sólo cortesía o charla sin pedir información académica (subtareas vacías).
"tramite" implica fechas o plazos. "escalamiento" es un reclamo, un caso urgente o algo que
requiere a un humano. Ordena las subtareas de mayor a menor prioridad (máximo 4)."""

PROMPT_AGENTE = """Eres un asistente de soporte académico para estudiantes universitarios.
Responde en español, claro y breve. Reglas:
1. Antes de afirmar cualquier norma, requisito o plazo usa buscar_reglamento y cita el documento fuente.
2. Si el estudiante pregunta por fechas límite usa calcular_plazo (pide la fecha de inicio si falta).
3. Si buscar_reglamento devuelve SIN_RESULTADOS, reformula la búsqueda una vez; si persiste, usa crear_ticket.
4. Si un plazo está VENCIDO o URGENTE, o el caso es urgente o requiere un humano, usa crear_ticket con prioridad alta.
5. Si el estudiante da un dato estable sobre sí mismo (carrera, semestre, beca), guárdalo con guardar_memoria.
6. Nunca inventes normas. Si no sabes, dilo y ofrece derivar con un ticket.
Sigue el plan en orden de prioridad. Cuando termines, responde al estudiante."""

PROMPT_DIRECTO = ("Eres un asistente de soporte académico amable. Responde en español, en una o dos "
                  "frases. No inventes información académica; si piden algo concreto, invítalos a preguntar.")


def _extraer_json(texto: str) -> dict:
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _texto(msg: BaseMessage) -> str:
    c = msg.content
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))


def _ultimo_humano(state: Estado) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return _texto(m)
    return ""


def construir_grafo(llm: Any, checkpointer: Any = None):
    """llm: cualquier ChatModel de LangChain con bind_tools (se inyecta en los tests)."""
    llm_tools = llm.bind_tools(TOOLS)

    def recuperar_contexto(state: Estado, config: RunnableConfig) -> dict:
        user_id = config.get("configurable", {}).get("user_id", "anonimo")
        recuerdos = get_memoria().recordar(user_id, _ultimo_humano(state))
        return {"memorias": "\n".join(f"- {r['texto']}" for r in recuerdos)}

    def planificar(state: Estado) -> dict:
        resp = llm.invoke([SystemMessage(content=PROMPT_PLANIFICADOR),
                           HumanMessage(content=_ultimo_humano(state))])
        plan = _extraer_json(_texto(resp))
        if plan.get("intencion") not in {"saludo", "consulta", "tramite", "escalamiento"}:
            plan = {"intencion": "consulta", "urgencia": "media", "subtareas": []}  # plan por defecto
        return {"plan": plan}

    def enrutar_plan(state: Estado) -> str:
        return "responder_directo" if state["plan"]["intencion"] == "saludo" else "agente"

    def _contexto_sistema(state: Estado, base: str) -> str:
        partes = [base]
        if state.get("resumen"):
            partes.append(f"Resumen de la conversación anterior:\n{state['resumen']}")
        if state.get("memorias"):
            partes.append(f"Datos recordados sobre el estudiante:\n{state['memorias']}")
        return "\n\n".join(partes)

    def agente(state: Estado) -> dict:
        plan = state.get("plan", {})
        sistema = _contexto_sistema(state, PROMPT_AGENTE) + (
            f"\n\nPlan de este turno (urgencia {plan.get('urgencia', 'media')}):\n"
            + json.dumps(plan.get("subtareas", []), ensure_ascii=False))
        return {"messages": [llm_tools.invoke([SystemMessage(content=sistema)] + state["messages"])]}

    def responder_directo(state: Estado) -> dict:
        sistema = _contexto_sistema(state, PROMPT_DIRECTO)
        return {"messages": [llm.invoke([SystemMessage(content=sistema)] + state["messages"])]}

    def despues_de_agente(state: Estado) -> str:
        ultimo = state["messages"][-1]
        return "tools" if getattr(ultimo, "tool_calls", None) else "resumir"

    def resumir(state: Estado) -> dict:
        """Si el historial es largo, resume lo antiguo y deja sólo los últimos mensajes."""
        msgs = state["messages"]
        if len(msgs) <= cfg.MAX_MESSAGES_BEFORE_SUMMARY:
            return {}
        # el corte debe caer al inicio de un turno (HumanMessage) para no separar tool_calls de sus resultados
        corte = next((i for i in range(len(msgs) - cfg.KEEP_LAST_MESSAGES, len(msgs))
                      if isinstance(msgs[i], HumanMessage)), None)
        if not corte:
            return {}
        antiguos = msgs[:corte]
        transcripcion = "\n".join(f"{m.type}: {_texto(m)}" for m in antiguos if _texto(m))
        prompt = ("Resume en máximo 120 palabras esta conversación de soporte académico, conservando "
                  "datos del estudiante, trámites, plazos y tickets creados.\n"
                  f"Resumen previo: {state.get('resumen', '(ninguno)')}\n\nConversación:\n{transcripcion}")
        nuevo = _texto(llm.invoke([HumanMessage(content=prompt)]))
        return {"resumen": nuevo, "messages": [RemoveMessage(id=m.id) for m in antiguos]}

    g = StateGraph(Estado)
    g.add_node("recuperar_contexto", recuperar_contexto)
    g.add_node("planificar", planificar)
    g.add_node("agente", agente)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("responder_directo", responder_directo)
    g.add_node("resumir", resumir)

    g.add_edge(START, "recuperar_contexto")
    g.add_edge("recuperar_contexto", "planificar")
    g.add_conditional_edges("planificar", enrutar_plan, ["agente", "responder_directo"])
    g.add_conditional_edges("agente", despues_de_agente, ["tools", "resumir"])
    g.add_edge("tools", "agente")
    g.add_edge("responder_directo", "resumir")
    g.add_edge("resumir", END)
    return g.compile(checkpointer=checkpointer)


def crear_checkpointer(path=cfg.DB_CHECKPOINTS) -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))

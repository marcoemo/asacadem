"""Ejecuta un turno de conversación mostrando las decisiones del agente (útil para la demo)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage


def _corto(texto: str, n: int = 160) -> str:
    texto = " ".join(str(texto).split())
    return texto if len(texto) <= n else texto[: n - 1] + "…"


def ejecutar_turno(grafo, texto: str, user_id: str, thread_id: str, verbose: bool = True) -> str:
    cfg = {"configurable": {"user_id": user_id, "thread_id": thread_id}, "recursion_limit": 20}
    respuesta = ""
    for paso in grafo.stream({"messages": [HumanMessage(content=texto)]}, cfg, stream_mode="updates"):
        for nodo, upd in paso.items():
            if not upd:
                continue
            if nodo == "recuperar_contexto" and upd.get("memorias") and verbose:
                print(f"  [memoria largo plazo] {_corto(upd['memorias'])}")
            elif nodo == "planificar" and verbose:
                p = upd["plan"]
                tareas = "; ".join(f"{t.get('orden')}.{t.get('tarea')} ({t.get('prioridad')})"
                                   for t in p.get("subtareas", []))
                print(f"  [plan] intención={p.get('intencion')} urgencia={p.get('urgencia')} {tareas}")
            elif nodo in ("agente", "responder_directo"):
                msg = upd["messages"][-1]
                for tc in getattr(msg, "tool_calls", []) or []:
                    if verbose:
                        print(f"  [tool] {tc['name']}({_corto(tc['args'], 100)})")
                if not getattr(msg, "tool_calls", None):
                    respuesta = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif nodo == "tools" and verbose:
                for m in upd["messages"]:
                    print(f"  [resultado] {_corto(m.content)}")
            elif nodo == "resumir" and upd.get("resumen") and verbose:
                print(f"  [contexto] historial resumido: {_corto(upd['resumen'])}")
    return respuesta

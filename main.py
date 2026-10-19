"""CLI del agente de soporte académico.

Uso:
  python main.py                       # chat interactivo
  python main.py --user ana --thread t2
  python main.py --reindex             # reconstruye el índice RAG
Comandos dentro del chat: /nuevo (nueva conversación), /memoria, /tickets, /salir
"""
import argparse
import uuid

from agente.grafo import construir_grafo, crear_checkpointer
from agente.llm import get_llm
from agente.runner import ejecutar_turno
from agente.servicios import get_memoria, get_rag, get_tickets_db


def main() -> None:
    ap = argparse.ArgumentParser(description="Agente de soporte académico")
    ap.add_argument("--user", default="alumno1", help="id del estudiante (memoria de largo plazo)")
    ap.add_argument("--thread", default="sesion-1", help="id de la conversación (memoria de corto plazo)")
    ap.add_argument("--reindex", action="store_true", help="reconstruir el índice RAG y salir")
    ap.add_argument("--quiet", action="store_true", help="no mostrar plan ni tools")
    args = ap.parse_args()

    if args.reindex:
        print(f"Índice reconstruido: {get_rag().reindexar()} fragmentos "
              f"({get_rag().embedder.name})")
        return

    grafo = construir_grafo(get_llm(), checkpointer=crear_checkpointer())
    thread = args.thread
    print(f"Agente listo (usuario={args.user}, conversación={thread}, embeddings={get_rag().embedder.name}).")
    print("Escribe tu consulta. Comandos: /nuevo /memoria /tickets /salir\n")
    while True:
        try:
            texto = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not texto:
            continue
        if texto == "/salir":
            break
        if texto == "/nuevo":
            thread = f"sesion-{uuid.uuid4().hex[:6]}"
            print(f"Nueva conversación: {thread} (la memoria de largo plazo se mantiene)\n")
            continue
        if texto == "/memoria":
            print("\n".join(f"- {m}" for m in get_memoria().todas(args.user)) or "(vacía)", "\n")
            continue
        if texto == "/tickets":
            filas = get_tickets_db().execute(
                "SELECT id, categoria, prioridad, estado, resumen FROM tickets WHERE usuario_id=?",
                (args.user,)).fetchall()
            print("\n".join(f"#{i} [{c}/{p}/{e}] {r}" for i, c, p, e, r in filas) or "(sin tickets)", "\n")
            continue
        respuesta = ejecutar_turno(grafo, texto, args.user, thread, verbose=not args.quiet)
        print(f"\nAgente: {respuesta}\n")


if __name__ == "__main__":
    main()

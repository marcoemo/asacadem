"""Ejecuta el conjunto de consultas de evaluación contra el agente REAL y registra métricas.

Uso:
  python evaluar.py                     # todas las consultas, 2 repeticiones
  python evaluar.py --rep 3             # 3 repeticiones (mide consistencia; gasta más API)
  python evaluar.py --escenario plazo   # sólo un escenario
  python evaluar.py --limite 5          # sólo las primeras 5 consultas (prueba rápida)
Cada turno agrega una línea a storage/logs/eventos.jsonl, que lee dashboard.py.
OJO: cada consulta usa varias llamadas al modelo. Revisa tu saldo antes de repetir muchas veces.
"""
import argparse
import json
import uuid
from datetime import date
from pathlib import Path

from agente import metricas
from agente.grafo import construir_grafo, crear_checkpointer
from agente.llm import get_llm
from agente.observabilidad import medir_turno

RUTA = Path(__file__).parent / "data" / "evaluacion" / "consultas.json"


def cargar_consultas(escenario: str | None = None, limite: int | None = None) -> list[dict]:
    consultas = json.loads(RUTA.read_text(encoding="utf-8"))
    for c in consultas:
        c["consulta"] = c["consulta"].replace("{hoy}", date.today().isoformat())
    if escenario:
        consultas = [c for c in consultas if c["escenario"] == escenario]
    return consultas[:limite] if limite else consultas


def ejecutar(grafo, consultas: list[dict], repeticiones: int) -> None:
    total = len(consultas) * repeticiones
    n = 0
    for rep in range(repeticiones):
        for c in consultas:
            n += 1
            ev = medir_turno(grafo, c["consulta"], f"eval-{c['id']}-{rep}-{uuid.uuid4().hex[:4]}",
                             f"hilo-{uuid.uuid4().hex[:8]}", escenario=c["escenario"], consulta_id=c["id"],
                             repeticion=rep, esperado=c["esperado"])
            marca = "OK " if ev.get("correcto") else "MAL"
            print(f"[{n}/{total}] {marca} {c['id']:>3} {ev['latencia_ms']:>8.0f} ms  ruta={ev['ruta']:<9} "
                  f"tools={[t['nombre'] for t in ev['tools']]}" + (f"  ERROR={ev['excepcion']}" if ev["excepcion"] else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluación del agente con consultas variadas")
    ap.add_argument("--rep", type=int, default=2, help="repeticiones por consulta (para consistencia)")
    ap.add_argument("--escenario", help="normal, ortografia, plazo, fuera_dominio, ambigua, multi_intencion, saludo, datos_personales, inyeccion")
    ap.add_argument("--limite", type=int, help="máximo de consultas")
    args = ap.parse_args()
    consultas = cargar_consultas(args.escenario, args.limite)
    print(f"{len(consultas)} consultas x {args.rep} repeticiones = {len(consultas) * args.rep} turnos\n")
    grafo = construir_grafo(get_llm(), checkpointer=crear_checkpointer(":memory:"))
    ejecutar(grafo, consultas, args.rep)
    print("\nResumen:", metricas.kpis(metricas.cargar_eventos()))
    print("Abre el dashboard con:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()

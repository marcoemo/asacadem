"""Demo guionada para la presentación (consume unos pocos centavos de la API).

Muestra: ruteo adaptativo (saludo vs consulta), RAG, cálculo de plazos, escalamiento a ticket,
y memoria de largo plazo entre dos conversaciones distintas.
Uso: python demo.py
"""
import uuid

from agente.grafo import construir_grafo, crear_checkpointer
from agente.llm import get_llm
from agente.runner import ejecutar_turno

USER = f"demo-{uuid.uuid4().hex[:5]}"  # estudiante nuevo en cada ejecución

ESCENARIOS = [
    ("1. Saludo -> ruta directa (sin tools)", "hilo-1", "¡Hola! Buenas tardes"),
    ("2. Datos del estudiante -> guarda memoria de largo plazo", "hilo-1",
     "Soy estudiante de Ingeniería en Informática, voy en 7º semestre y tengo la Beca de Alimentación."),
    ("3. Consulta normativa -> RAG con fuente", "hilo-1", "¿Qué necesito para titularme?"),
    ("4. Plazo VENCIDO -> calcula y escala con ticket de prioridad alta", "hilo-1",
     "Me publicaron una nota el 2026-09-01 y quiero apelar, ¿llego a tiempo?"),
    ("5. Fuera de dominio -> reformula y deriva", "hilo-1",
     "¿Cómo configuro la contraseña del wifi de la sala 204?"),
    ("6. NUEVA conversación, mismo estudiante -> recuerda por memoria de largo plazo", "hilo-2",
     "¿Recuerdas qué carrera estudio y qué beca tengo?"),
]


def main() -> None:
    grafo = construir_grafo(get_llm(), checkpointer=crear_checkpointer(":memory:"))
    for titulo, hilo, texto in ESCENARIOS:
        print(f"\n=== {titulo} ===\nTú: {texto}")
        print(f"Agente: {ejecutar_turno(grafo, texto, USER, hilo)}")


if __name__ == "__main__":
    main()

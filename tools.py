"""Herramientas del agente (consulta, escritura y razonamiento).

El LLM elige qué herramienta usar leyendo el nombre, el docstring y el esquema de
argumentos de cada una (function calling). Para agregar una tool nueva basta con
definirla con @tool y sumarla a la lista TOOLS al final de este archivo.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from . import config
from .servicios import get_memoria, get_rag, get_tickets_db

PRIORIDADES = {"alta": "4 horas hábiles", "media": "24 horas hábiles", "baja": "72 horas hábiles"}
CATEGORIAS = ["titulacion", "becas", "notas", "inscripcion", "certificados", "plataforma", "otro"]


def _user_id(config_: RunnableConfig | None) -> str:
    return ((config_ or {}).get("configurable") or {}).get("user_id", "anonimo")


# ---------------------------------------------------------------- CONSULTA
@tool
def buscar_reglamento(consulta: str) -> str:
    """Busca en el reglamento y los documentos académicos de la institución (titulación,
    becas, notas, certificados, prácticas, malla). Úsala SIEMPRE antes de afirmar
    cualquier norma o requisito. Devuelve fragmentos con su fuente."""
    resultados = get_rag().buscar_relevantes(consulta)
    if not resultados:
        return ("SIN_RESULTADOS: no hay información relevante en los documentos para esa consulta. "
                "Reformula la búsqueda una vez; si sigue sin resultados, deriva con crear_ticket.")
    return "\n\n".join(
        f"[{r['fuente']} > {r['seccion']}] (relevancia {r['score']:.2f})\n{r['texto']}"
        for r in resultados
    )


@tool
def consultar_tickets(config_: RunnableConfig) -> str:
    """Lista los tickets de soporte que ya creó este estudiante y su estado."""
    filas = get_tickets_db().execute(
        "SELECT id, categoria, prioridad, estado, resumen FROM tickets WHERE usuario_id=? ORDER BY id",
        (_user_id(config_),),
    ).fetchall()
    if not filas:
        return "El estudiante no tiene tickets registrados."
    return "\n".join(f"#{i} [{c}/{p}/{e}] {r}" for i, c, p, e, r in filas)


@tool
def recordar(consulta: str, config_: RunnableConfig) -> str:
    """Busca en la memoria de largo plazo datos guardados antes sobre este estudiante
    (carrera, semestre, beneficios, trámites anteriores)."""
    res = get_memoria().recordar(_user_id(config_), consulta)
    return "\n".join(f"- {r['texto']}" for r in res) if res else "No hay recuerdos relevantes."


# ---------------------------------------------------------------- ESCRITURA
@tool
def crear_ticket(resumen: str, categoria: str, prioridad: str, config_: RunnableConfig) -> str:
    """Crea un ticket para derivar el caso a un ejecutivo humano de Secretaría Académica.
    Úsala cuando no haya información en los documentos, cuando haya un plazo vencido o por
    vencer, o cuando el caso sea urgente. categoria: titulacion, becas, notas, inscripcion,
    certificados, plataforma u otro. prioridad: alta (urgente/plazo en riesgo), media o baja."""
    prioridad, categoria = prioridad.lower().strip(), categoria.lower().strip()
    if prioridad not in PRIORIDADES:
        return f"ERROR: prioridad inválida '{prioridad}'. Usa: {', '.join(PRIORIDADES)}."
    if categoria not in CATEGORIAS:
        categoria = "otro"
    db = get_tickets_db()
    cur = db.execute(
        "INSERT INTO tickets (usuario_id, categoria, prioridad, resumen, creado) VALUES (?,?,?,?,?)",
        (_user_id(config_), categoria, prioridad, resumen.strip(), datetime.now().isoformat()),
    )
    db.commit()
    return (f"Ticket #{cur.lastrowid} creado (categoría {categoria}, prioridad {prioridad}). "
            f"Tiempo de respuesta estimado: {PRIORIDADES[prioridad]}.")


@tool
def guardar_memoria(dato: str, config_: RunnableConfig) -> str:
    """Guarda en la memoria de largo plazo un dato estable y útil del estudiante
    (carrera, semestre, beca que tiene, trámite en curso). No guardes datos sensibles
    como RUT, contraseñas, salud o situación económica detallada."""
    return get_memoria().guardar(_user_id(config_), dato)


# ---------------------------------------------------------------- RAZONAMIENTO
def _cargar_plazos() -> dict:
    return json.loads(config.PLAZOS_PATH.read_text(encoding="utf-8"))


@tool
def calcular_plazo(tramite: str, fecha_inicio: str = "") -> str:
    """Calcula la fecha límite de un trámite en días hábiles y dice cuánto queda.
    tramite: inscripcion_titulacion, apelacion_nota, retiro_asignatura,
    postulacion_beca_excelencia o certificado_alumno. fecha_inicio: YYYY-MM-DD del hito
    desde el cual corre el plazo (si se omite se usa hoy)."""
    datos = _cargar_plazos()
    tramites = datos["tramites"]
    if tramite not in tramites:
        return f"ERROR: trámite desconocido '{tramite}'. Opciones: {', '.join(tramites)}."
    try:
        inicio = date.fromisoformat(fecha_inicio) if fecha_inicio else date.today()
    except ValueError:
        return "ERROR: fecha_inicio debe tener formato YYYY-MM-DD."

    t, feriados = tramites[tramite], datos.get("feriados", [])
    limite = np.busday_offset(inicio, t["dias_habiles"], roll="forward", holidays=feriados)
    limite = date.fromisoformat(str(limite))
    hoy = date.today()
    if limite < hoy:
        restantes = -int(np.busday_count(limite, hoy, holidays=feriados))
    else:
        restantes = int(np.busday_count(hoy, limite, holidays=feriados))

    if restantes < 0:
        estado = f"VENCIDO hace {-restantes} días hábiles"
    elif restantes <= 2:
        estado = f"URGENTE: quedan {restantes} días hábiles"
    else:
        estado = f"VIGENTE: quedan {restantes} días hábiles"
    return (f"{t['nombre']}: {t['dias_habiles']} días hábiles desde {t['desde']} "
            f"({inicio.isoformat()}). Fecha límite: {limite.isoformat()}. Estado: {estado}. "
            f"Fuente: {t['fuente']}.")


TOOLS = [buscar_reglamento, calcular_plazo, crear_ticket, consultar_tickets, guardar_memoria, recordar]

"""Pruebas del grafo con un LLM guionado (sin red). Verifican las DECISIONES del agente:
ruteo, elección de tools, escalamiento, memoria de corto/largo plazo y resumen de contexto."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agente import config as cfg
from agente.grafo import construir_grafo, crear_checkpointer
from agente.servicios import get_memoria, get_tickets_db
from tests.conftest import ia


def plan(intencion, urgencia="media", subtareas=None):
    return ia(json.dumps({"intencion": intencion, "urgencia": urgencia, "subtareas": subtareas or []}))


def correr(grafo, texto, user="alumno", thread="t1"):
    return grafo.invoke({"messages": [HumanMessage(content=texto)]},
                        config={"configurable": {"user_id": user, "thread_id": thread}, "recursion_limit": 20})


def herramientas_usadas(res):
    return [tc["name"] for m in res["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]


def test_saludo_va_por_ruta_directa_sin_tools(llm_guionado):
    llm = llm_guionado(plan("saludo"), ia("¡Hola! ¿En qué te ayudo?"))
    res = correr(construir_grafo(llm), "hola!")
    assert herramientas_usadas(res) == []
    assert res["messages"][-1].content.startswith("¡Hola")


def test_consulta_usa_rag_y_responde(llm_guionado):
    llm = llm_guionado(
        plan("consulta", subtareas=[{"orden": 1, "tarea": "buscar requisitos", "prioridad": "alta"}]),
        ia(tool="buscar_reglamento", args={"consulta": "requisitos para titularme"}),
        ia("Necesitas el 100% de la malla aprobada (reglamento_titulacion.md)."),
    )
    res = correr(construir_grafo(llm), "¿Qué necesito para titularme?")
    assert herramientas_usadas(res) == ["buscar_reglamento"]
    tool_msg = next(m for m in res["messages"] if isinstance(m, ToolMessage))
    assert "reglamento_titulacion.md" in tool_msg.content


def test_escalamiento_crea_ticket_en_base_de_datos(llm_guionado):
    llm = llm_guionado(
        plan("escalamiento", "alta"),
        ia(tool="crear_ticket", args={"resumen": "Plazo de apelación vencido", "categoria": "notas",
                                      "prioridad": "alta"}),
        ia("Creé el ticket #1, un ejecutivo te contactará."),
    )
    res = correr(construir_grafo(llm), "Se me pasó el plazo de apelación, ayuda", user="est-9")
    assert "crear_ticket" in herramientas_usadas(res)
    filas = get_tickets_db().execute("SELECT prioridad, usuario_id FROM tickets").fetchall()
    assert ("alta", "est-9") in filas


def test_adaptacion_sin_resultados_reformula_y_luego_deriva(llm_guionado):
    llm = llm_guionado(
        plan("consulta"),
        ia(tool="buscar_reglamento", args={"consulta": "contraseña wifi"}, id_="c1"),
        ia(tool="buscar_reglamento", args={"consulta": "acceso internet campus"}, id_="c2"),
        ia(tool="crear_ticket", args={"resumen": "Consulta de wifi sin documentación",
                                      "categoria": "plataforma", "prioridad": "baja"}, id_="c3"),
        ia("No tengo esa información; dejé un ticket para que te ayuden."),
    )
    res = correr(construir_grafo(llm), "¿Cómo cambio mi contraseña del wifi?")
    assert herramientas_usadas(res) == ["buscar_reglamento", "buscar_reglamento", "crear_ticket"]


def test_plan_invalido_usa_plan_por_defecto(llm_guionado):
    llm = llm_guionado(ia("esto no es json"), ia("Respuesta de todos modos."))
    res = correr(construir_grafo(llm), "hola, una duda")
    assert res["plan"]["intencion"] == "consulta"


def test_memoria_corto_plazo_persiste_en_sqlite(llm_guionado, tmp_path):
    cp = crear_checkpointer(tmp_path / "cp.db")
    llm = llm_guionado(plan("saludo"), ia("Hola Ana"), plan("saludo"), ia("Te llamas Ana."))
    grafo = construir_grafo(llm, checkpointer=cp)
    correr(grafo, "Hola, soy Ana", thread="hilo-A")
    res = correr(grafo, "¿cómo me llamo?", thread="hilo-A")
    assert len(res["messages"]) == 4  # el historial del hilo se conservó
    otro = correr(construir_grafo(llm_guionado(plan("saludo"), ia("Hola")), checkpointer=cp),
                  "hola", thread="hilo-B")
    assert len(otro["messages"]) == 2  # otro hilo parte limpio


def test_memoria_largo_plazo_se_inyecta_en_nuevo_hilo(llm_guionado):
    get_memoria().guardar("lore", "Estudia Ingeniería en Informática, 7º semestre")
    llm = llm_guionado(plan("saludo"), ia("Hola, de Informática"))
    correr(construir_grafo(llm), "qué estudio yo, estudia carrera", user="lore", thread="nuevo")
    sistema = llm.llamadas[-1][0].content
    assert "Informática" in sistema


def test_resumen_de_contexto_recorta_historial(llm_guionado, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_MESSAGES_BEFORE_SUMMARY", 4)
    monkeypatch.setattr(cfg, "KEEP_LAST_MESSAGES", 2)
    respuestas = []
    for i in range(3):
        respuestas += [plan("saludo"), ia(f"resp {i}")]
    respuestas.append(ia("Resumen: el estudiante saludó varias veces."))  # llamada de resumen (turno 3)
    llm = llm_guionado(*respuestas)
    grafo = construir_grafo(llm, checkpointer=crear_checkpointer(":memory:"))
    for i in range(3):
        res = correr(grafo, f"mensaje {i}", thread="largo")
    assert "Resumen" in res["resumen"]
    assert len(res["messages"]) <= 4

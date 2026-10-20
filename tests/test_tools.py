from datetime import date, timedelta

from agente.servicios import get_memoria
from agente.tools import (calcular_plazo, consultar_tickets, crear_ticket, guardar_memoria,
                          recordar)

CFG = {"configurable": {"user_id": "est-test"}}


def test_plazo_vigente():
    hoy = date.today()
    out = calcular_plazo.invoke({"tramite": "apelacion_nota", "fecha_inicio": hoy.isoformat()})
    assert "Fecha límite" in out and ("VIGENTE" in out or "URGENTE" in out)


def test_plazo_vencido():
    inicio = (date.today() - timedelta(days=60)).isoformat()
    assert "VENCIDO" in calcular_plazo.invoke({"tramite": "apelacion_nota", "fecha_inicio": inicio})


def test_plazo_dias_habiles_exactos():
    # lunes 2025-03-03 + 5 días hábiles = lunes 2025-03-10
    out = calcular_plazo.invoke({"tramite": "apelacion_nota", "fecha_inicio": "2025-03-03"})
    assert "2025-03-10" in out


def test_plazo_tramite_invalido_lista_opciones():
    out = calcular_plazo.invoke({"tramite": "inventado"})
    assert out.startswith("ERROR") and "apelacion_nota" in out


def test_plazo_fecha_invalida():
    assert "ERROR" in calcular_plazo.invoke({"tramite": "apelacion_nota", "fecha_inicio": "ayer"})


def test_crear_y_consultar_ticket():
    out = crear_ticket.invoke({"resumen": "No puedo inscribir ramos", "categoria": "inscripcion",
                               "prioridad": "alta"}, config=CFG)
    assert "Ticket #" in out and "4 horas" in out
    assert "No puedo inscribir ramos" in consultar_tickets.invoke({}, config=CFG)


def test_ticket_prioridad_invalida():
    out = crear_ticket.invoke({"resumen": "x", "categoria": "otro", "prioridad": "urgentisima"}, config=CFG)
    assert out.startswith("ERROR")


def test_tickets_aislados_por_usuario():
    crear_ticket.invoke({"resumen": "caso A", "categoria": "otro", "prioridad": "baja"}, config=CFG)
    otro = {"configurable": {"user_id": "otro-estudiante"}}
    assert "no tiene tickets" in consultar_tickets.invoke({}, config=otro)


def test_memoria_largo_plazo_guarda_y_recupera():
    guardar_memoria.invoke({"dato": "Estudia Ingeniería en Informática, 7º semestre"}, config=CFG)
    guardar_memoria.invoke({"dato": "Tiene la Beca de Alimentación"}, config=CFG)
    out = recordar.invoke({"consulta": "qué carrera estudia"}, config=CFG)
    assert "Informática" in out


def test_memoria_no_duplica():
    m = get_memoria()
    m.guardar("u1", "Estudia Ingeniería en Informática")
    assert m.guardar("u1", "Estudia Ingeniería en Informática").startswith("Ya estaba")
    assert len(m.todas("u1")) == 1

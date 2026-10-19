from agente.rag import RagIndex
from agente.tools import buscar_reglamento


def test_indexa_documentos():
    assert len(RagIndex().chunks) >= 15


def test_recupera_seccion_correcta():
    fuentes = [r["seccion"] for r in RagIndex().buscar("beca de alimentación requisitos", k=2)]
    assert "Beca de Alimentación" in fuentes


def test_apelacion_de_notas():
    top = RagIndex().buscar("cuántos días tengo para apelar una nota", k=1)[0]
    assert top["fuente"] == "procesos_academicos.md"


def test_fuera_de_dominio_devuelve_sin_resultados():
    assert "SIN_RESULTADOS" in buscar_reglamento.invoke({"consulta": "receta de pizza napolitana"})


def test_tool_incluye_fuente():
    out = buscar_reglamento.invoke({"consulta": "requisitos para titularme"})
    assert "reglamento_titulacion.md" in out

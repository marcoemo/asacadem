# Evidencia de pruebas

## 1. Pruebas automáticas (sin red ni costo)
Se ejecutan con `python -m pytest -v`. El grafo se prueba con un **LLM guionado** (`tests/conftest.py`), que devuelve respuestas predefinidas; así se verifica de forma determinista *qué decisiones toma la orquestación* (ruteo, tools, escalamiento, memoria, resumen) sin depender del modelo real.

| Prueba | Qué demuestra | Indicador |
|---|---|---|
| `test_saludo_va_por_ruta_directa_sin_tools` | Ruteo adaptativo: saludo no usa tools | IE5, IE7 |
| `test_consulta_usa_rag_y_responde` | El agente consulta el RAG y la tool devuelve la fuente | IE1, IE4 |
| `test_escalamiento_crea_ticket_en_base_de_datos` | Escritura real en SQLite con prioridad alta | IE1 |
| `test_adaptacion_sin_resultados_reformula_y_luego_deriva` | Secuencia adaptativa: busca, reformula, deriva con ticket | IE5, IE7 |
| `test_plan_invalido_usa_plan_por_defecto` | Robustez ante un plan mal formado | IE5 |
| `test_memoria_corto_plazo_persiste_en_sqlite` | Historial por `thread_id` persistente; hilos aislados | IE3 |
| `test_memoria_largo_plazo_se_inyecta_en_nuevo_hilo` | Recuerdo semántico entre conversaciones | IE3, IE4 |
| `test_resumen_de_contexto_recorta_historial` | Gestión de ventana de contexto | IE3 |
| `test_rag.py` (5 pruebas) | Indexación, recuperación correcta, umbral de relevancia, citas de fuente | IE4 |
| `test_tools.py` (10 pruebas) | Días hábiles exactos, plazos vencidos/vigentes, validación de entradas, aislamiento por usuario, no duplicar memorias | IE1 |

### Resultado de la última ejecución
```
collecting ... collected 23 items

tests/test_grafo.py::test_saludo_va_por_ruta_directa_sin_tools PASSED    [  4%]
tests/test_grafo.py::test_consulta_usa_rag_y_responde PASSED             [  8%]
tests/test_grafo.py::test_escalamiento_crea_ticket_en_base_de_datos PASSED [ 13%]
tests/test_grafo.py::test_adaptacion_sin_resultados_reformula_y_luego_deriva PASSED [ 17%]
tests/test_grafo.py::test_plan_invalido_usa_plan_por_defecto PASSED      [ 21%]
tests/test_grafo.py::test_memoria_corto_plazo_persiste_en_sqlite PASSED  [ 26%]
tests/test_grafo.py::test_memoria_largo_plazo_se_inyecta_en_nuevo_hilo PASSED [ 30%]
tests/test_grafo.py::test_resumen_de_contexto_recorta_historial PASSED   [ 34%]
tests/test_rag.py::test_indexa_documentos PASSED                         [ 39%]
tests/test_rag.py::test_recupera_seccion_correcta PASSED                 [ 43%]
tests/test_rag.py::test_apelacion_de_notas PASSED                        [ 47%]
tests/test_rag.py::test_fuera_de_dominio_devuelve_sin_resultados PASSED  [ 52%]
tests/test_rag.py::test_tool_incluye_fuente PASSED                       [ 56%]
tests/test_tools.py::test_plazo_vigente PASSED                           [ 60%]
tests/test_tools.py::test_plazo_vencido PASSED                           [ 65%]
tests/test_tools.py::test_plazo_dias_habiles_exactos PASSED              [ 69%]
tests/test_tools.py::test_plazo_tramite_invalido_lista_opciones PASSED   [ 73%]
tests/test_tools.py::test_plazo_fecha_invalida PASSED                    [ 78%]
tests/test_tools.py::test_crear_y_consultar_ticket PASSED                [ 82%]
tests/test_tools.py::test_ticket_prioridad_invalida PASSED               [ 86%]
tests/test_tools.py::test_tickets_aislados_por_usuario PASSED            [ 91%]
tests/test_tools.py::test_memoria_largo_plazo_guarda_y_recupera PASSED   [ 95%]
tests/test_tools.py::test_memoria_no_duplica PASSED                      [100%]

============================== 23 passed in 1.88s ==============================
```

## 2. Pruebas con el modelo real (`python demo.py`)
`demo.py` ejecuta 6 escenarios con la API de OpenRouter y muestra en consola el plan, las tools llamadas y sus resultados. Pegar aquí la salida de la ejecución del equipo como evidencia (capturas o texto):

| # | Escenario | Comportamiento esperado |
|---|---|---|
| 1 | Saludo | Plan `saludo` -> `responder_directo`, sin tools |
| 2 | El estudiante cuenta su carrera y beca | Llama `guardar_memoria` |
| 3 | "¿Qué necesito para titularme?" | `buscar_reglamento` y respuesta citando `reglamento_titulacion.md` |
| 4 | Apelar una nota publicada el 2026-09-01 | `calcular_plazo` -> VENCIDO -> `crear_ticket` prioridad alta |
| 5 | Pregunta fuera de los documentos (wifi) | `buscar_reglamento` -> SIN_RESULTADOS -> reformula -> ticket |
| 6 | Nueva conversación, mismo estudiante | Recuerda carrera y beca desde la memoria de largo plazo |

### Salida de la ejecución del equipo
_(pegar aquí)_

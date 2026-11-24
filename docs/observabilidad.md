# Observabilidad, trazabilidad y uso responsable (EP3)

Este documento describe **cómo se mide** el agente. Los resultados, hallazgos y recomendaciones se documentan en el informe técnico a partir de los datos reales que genere `evaluar.py`.

## Cómo se obtienen los datos
```bash
python evaluar.py --rep 3      # 30 consultas x 3 repeticiones contra el modelo real
streamlit run dashboard.py     # abre el dashboard con los logs
```
Cada turno agrega una línea JSON a `storage/logs/eventos.jsonl` (`agente/observabilidad.py`, función `medir_turno`). Un *callback* de LangChain mide el tiempo de cada nodo del grafo, cada llamada al modelo y cada tool, y suma los tokens que informa la API.

## Métricas
| Métrica | Definición | Cómo se mide | Dónde |
|---|---|---|---|
| Precisión | % de turnos cuyo comportamiento coincide con lo esperado (ruta correcta, tools requeridas usadas, tools prohibidas no usadas) | Cada consulta de `data/evaluacion/consultas.json` declara su `esperado` | `observabilidad.evaluar_esperado`, `metricas.precision_por_escenario` |
| Consistencia | Para una misma consulta repetida, % de repeticiones con la misma decisión que la más frecuente (misma ruta y mismo conjunto de tools) | Requiere `--rep 2` o más | `metricas.consistencia_por_consulta` |
| Frecuencia de errores | % de turnos con error de tool, excepción o plan inválido (JSON del planificador inválido). `SIN_RESULTADOS` y "comportamiento incorrecto" se reportan aparte | Estados de las tools y banderas del evento | `metricas.errores_por_tipo` |
| Latencia | Tiempo total del turno (p50 y p95), por nodo del grafo, por tool y por escenario | `time.perf_counter` en el callback | `metricas.latencia_por_nodo`, `latencia_por_tool`, `latencia_por_escenario` |
| Uso de recursos | Tokens de entrada y salida, llamadas al modelo por turno, costo estimado y memoria del proceso (MB) | `usage_metadata` de la respuesta, `psutil` | `metricas.kpis` |

## Escenarios de variabilidad (`data/evaluacion/consultas.json`)
30 consultas en 9 escenarios: normal, ortografía, plazos, fuera de dominio, ambiguas, varias intenciones, saludos, datos personales e inyección de prompt. El campo `esperado` de cada consulta es una **decisión de diseño del equipo**: revísenlo y ajústenlo si no coincide con lo que consideran correcto.

## Seguridad, trazabilidad y uso responsable
| Medida | Qué hace | Dónde |
|---|---|---|
| Minimización de datos personales | Antes de llegar al modelo y al historial, se reemplazan RUT, correos, teléfonos y tarjetas por marcas como `[RUT]` | Nodo `guardia`, `seguridad.redactar_pii` |
| Logs sin datos personales | Las entradas y salidas se redactan otra vez antes de escribirse | `observabilidad.medir_turno` |
| Memoria sin datos sensibles | `guardar_memoria` rechaza textos con datos personales | `tools.guardar_memoria` |
| Defensa contra inyección de prompts | Heurística de patrones; los mensajes sospechosos se bloquean sin llamar al modelo | `seguridad.detectar_inyeccion`, nodo `responder_bloqueo` |
| Límite de entrada | Mensajes de más de `MAX_INPUT_CHARS` (1500) se bloquean | Nodo `guardia` |
| Derecho de supresión | `/olvidar` borra recuerdos, tickets e historial de la conversación | `servicios.olvidar_usuario`, `main.py` |
| Transparencia | El agente se presenta como IA y los prompts le piden aclararlo | `main.py`, `PROMPT_AGENTE` |
| Trazabilidad | Cada turno queda registrado con su plan, tools, tiempos y resultado | `storage/logs/eventos.jsonl` |

## Limitaciones de la medición (decláralas en la defensa)
- La detección de inyección es heurística: se evade con otras frases. No es una defensa completa.
- La precisión depende de que el `esperado` de cada consulta esté bien definido.
- Con `temperature=0` el modelo es más estable, pero no totalmente determinista; la consistencia se mide, no se asume.
- Los tokens sólo se registran si la API los informa; el costo sólo se calcula si se definen `PRECIO_IN_USD_1M` y `PRECIO_OUT_USD_1M` en `.env`.
- La latencia incluye la red y la carga del proveedor, por lo que varía entre ejecuciones.
- Los datos personales se detectan con expresiones regulares (formatos chilenos de RUT y teléfono): no cubre nombres ni direcciones.

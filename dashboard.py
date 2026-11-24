"""Dashboard de observabilidad del agente (Streamlit).  Ejecutar:  streamlit run dashboard.py"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from agente import config, metricas

st.set_page_config(page_title="Observabilidad del agente", layout="wide")
st.title("Observabilidad del agente de soporte académico")

ruta = st.sidebar.text_input("Archivo de logs", os.getenv("AGENTE_LOG_PATH", str(config.LOG_PATH)))
df = metricas.cargar_eventos(ruta)
if df.empty:
    st.warning("No hay eventos todavía. Genera datos con `python evaluar.py` y recarga esta página.")
    st.stop()

escenarios = sorted(df["escenario"].dropna().unique())
sel = st.sidebar.multiselect("Escenarios", escenarios, default=escenarios)
df = df[df["escenario"].isin(sel)]
if df.empty:
    st.info("Selecciona al menos un escenario.")
    st.stop()
st.sidebar.caption(f"{len(df)} turnos · {df['ts'].min():%Y-%m-%d %H:%M} a {df['ts'].max():%Y-%m-%d %H:%M}")

k = metricas.kpis(df)
c = st.columns(6)
fmt = lambda v, u="": "s/d" if pd.isna(v) else f"{v}{u}"  # noqa: E731
c[0].metric("Turnos", k["turnos"])
c[1].metric("Precisión", fmt(k["precision_pct"], " %"))
c[2].metric("Consistencia", fmt(k["consistencia_pct"], " %"))
c[3].metric("Tasa de error", fmt(k["tasa_error_pct"], " %"))
c[4].metric("Latencia p50 / p95", f"{k['latencia_p50_ms']:.0f} / {k['latencia_p95_ms']:.0f} ms")
c[5].metric("Tokens por turno", f"{k['tokens_por_turno']:.0f}")

t1, t2, t3, t4, t5 = st.tabs(["Precisión y consistencia", "Latencia", "Recursos", "Errores y trazas", "Seguridad"])

with t1:
    a, b = st.columns(2)
    p = metricas.precision_por_escenario(df)
    if len(p):
        a.plotly_chart(px.bar(p, x="escenario", y="precision_pct", text="precision_pct", range_y=[0, 105],
                              title="Precisión por escenario (%)"), use_container_width=True)
    cs = metricas.consistencia_por_consulta(df)
    if len(cs):
        b.plotly_chart(px.bar(cs.sort_values("consistencia_pct"), x="consulta_id", y="consistencia_pct",
                              color="escenario", range_y=[0, 105],
                              title="Consistencia por consulta (misma decisión en las repeticiones, %)"),
                       use_container_width=True)
    else:
        b.info("Para medir consistencia ejecuta `python evaluar.py --rep 3` (varias repeticiones por consulta).")
    st.dataframe(p, use_container_width=True, hide_index=True)

with t2:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="latencia_ms", nbins=30, title="Distribución de la latencia por turno (ms)"),
                   use_container_width=True)
    n = metricas.latencia_por_nodo(df)
    b.plotly_chart(px.bar(n, x="nodo", y="media_ms", text="participacion_pct",
                          title="Latencia media por nodo del grafo (ms) y % del total"), use_container_width=True)
    a, b = st.columns(2)
    e = metricas.latencia_por_escenario(df)
    a.plotly_chart(px.bar(e, x="escenario", y="p95_ms", title="Latencia p95 por escenario (ms)"), use_container_width=True)
    b.plotly_chart(px.scatter(df, x="latencia_ms", y="llamadas_llm", color="ruta",
                              title="Latencia vs. número de llamadas al modelo"), use_container_width=True)
    st.subheader("Latencia por tool")
    st.dataframe(metricas.latencia_por_tool(df), use_container_width=True, hide_index=True)

with t3:
    a, b = st.columns(2)
    d = df.assign(tokens=df["tokens_in"] + df["tokens_out"])
    a.plotly_chart(px.box(d, x="escenario", y="tokens", title="Tokens por turno según escenario"), use_container_width=True)
    b.plotly_chart(px.scatter(d, x="tokens", y="latencia_ms", color="escenario",
                              title="Tokens vs. latencia"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.line(df.sort_values("ts"), x="ts", y="memoria_mb", title="Memoria del proceso (MB)"), use_container_width=True)
    b.metric("Costo total estimado (USD)", k["costo_total_usd"],
             help="Define PRECIO_IN_USD_1M y PRECIO_OUT_USD_1M en .env para calcularlo.")
    b.metric("Llamadas al modelo por turno", k["llamadas_llm_por_turno"])

with t4:
    er = metricas.errores_por_tipo(df)
    st.plotly_chart(px.bar(er, x="tipo", y="turnos", title="Turnos afectados por tipo de problema"), use_container_width=True)
    st.subheader("Trazas (turnos con problemas primero)")
    sol = st.checkbox("Mostrar sólo turnos incorrectos o con error", value=True)
    t = df.copy()
    t["incorrecto"] = t["correcto"].eq(False) if "correcto" in t else False
    if sol:
        t = t[t["incorrecto"] | t["tiene_error"] | (t["sin_resultados"] > 0)]
    cols = [x for x in ["ts", "consulta_id", "escenario", "entrada", "ruta", "tools_usadas", "latencia_ms",
                        "errores_tool", "sin_resultados", "plan_fallback", "excepcion", "correcto"] if x in t]
    st.dataframe(t.sort_values("latencia_ms", ascending=False)[cols].astype({"tools_usadas": str}),
                 use_container_width=True, hide_index=True)
    if len(t):
        fila = st.selectbox("Detalle de una traza", t.index, format_func=lambda i: f"{t.loc[i, 'consulta_id']} · {t.loc[i, 'entrada'][:60]}")
        st.json({k_: (v if not isinstance(v, (pd.Timestamp, tuple)) else str(v)) for k_, v in t.loc[fila].to_dict().items()
                 if k_ in ("entrada", "salida", "ruta", "intencion", "urgencia", "tools", "latencia_nodos", "alerta", "pii", "excepcion")})

with t5:
    a, b, c3 = st.columns(3)
    a.metric("Turnos con datos personales redactados", int(df["pii"].apply(len).gt(0).sum()))
    b.metric("Turnos bloqueados por la guardia", int((df["ruta"] == "bloqueado").sum()))
    ino = df[df["escenario"] == "inyeccion"]
    c3.metric("Inyecciones bloqueadas", f"{int((ino['ruta'] == 'bloqueado').sum())} de {len(ino)}" if len(ino) else "s/d")
    st.caption("Los logs guardan los textos con datos personales ya redactados ([RUT], [EMAIL], [TELEFONO], [TARJETA]).")
    st.dataframe(df[df["alerta"].astype(str) != ""][["ts", "consulta_id", "entrada", "alerta", "ruta"]],
                 use_container_width=True, hide_index=True)

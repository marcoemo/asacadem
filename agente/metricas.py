"""Cálculo de métricas a partir de los eventos registrados (los usan el dashboard y evaluar.py).

Definiciones (ver docs/observabilidad.md):
- precisión: % de turnos con resultado esperado cuyo comportamiento fue correcto (ruta + tools).
- consistencia: para una misma consulta repetida, % de repeticiones que coinciden con la decisión más
  frecuente (misma ruta y mismo conjunto de tools).
- tasa de error: % de turnos con error de tool, excepción o plan inválido.
- latencia: tiempo total del turno (p50, p95); también por nodo y por tool.
- recursos: tokens, llamadas al LLM, costo estimado y memoria del proceso.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def cargar_eventos(path=None) -> pd.DataFrame:
    path = Path(path or config.LOG_PATH)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    filas = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(filas)
    df["ts"] = pd.to_datetime(df["ts"])
    df["tools_usadas"] = df["tools"].apply(lambda ts: tuple(sorted({t["nombre"] for t in ts})))
    df["tiene_error"] = (df["errores_tool"] > 0) | df["excepcion"].notna() | df["plan_fallback"]
    return df


def _pct(serie) -> float:
    return round(float(serie.mean()) * 100, 1) if len(serie) else float("nan")


def kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ev = df[df["correcto"].notna()] if "correcto" in df else df.iloc[0:0]
    return {
        "turnos": len(df),
        "precision_pct": _pct(ev["correcto"].astype(bool)) if len(ev) else float("nan"),
        "consistencia_pct": consistencia_global(df),
        "tasa_error_pct": _pct(df["tiene_error"]),
        "latencia_p50_ms": round(float(df["latencia_ms"].quantile(0.5)), 1),
        "latencia_p95_ms": round(float(df["latencia_ms"].quantile(0.95)), 1),
        "tokens_por_turno": round(float((df["tokens_in"] + df["tokens_out"]).mean()), 1),
        "llamadas_llm_por_turno": round(float(df["llamadas_llm"].mean()), 2),
        "costo_total_usd": round(float(df["costo_usd"].sum()), 4),
        "memoria_mb_max": round(float(df["memoria_mb"].max()), 1),
    }


def precision_por_escenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "correcto" not in df:
        return pd.DataFrame(columns=["escenario", "turnos", "precision_pct"])
    ev = df[df["correcto"].notna()].copy()
    ev["correcto"] = ev["correcto"].astype(bool)
    g = ev.groupby("escenario")["correcto"].agg(["count", "mean"]).reset_index()
    g.columns = ["escenario", "turnos", "precision_pct"]
    g["precision_pct"] = (g["precision_pct"] * 100).round(1)
    return g.sort_values("precision_pct")


def consistencia_por_consulta(df: pd.DataFrame) -> pd.DataFrame:
    """Sólo consultas con 2 o más repeticiones."""
    filas = []
    if df.empty or "consulta_id" not in df:
        return pd.DataFrame(columns=["consulta_id", "escenario", "repeticiones", "consistencia_pct"])
    for cid, g in df[df["consulta_id"].notna()].groupby("consulta_id"):
        if len(g) < 2:
            continue
        firmas = g.apply(lambda r: (r["ruta"], r["tools_usadas"]), axis=1)
        moda = firmas.value_counts().iloc[0]
        filas.append({"consulta_id": cid, "escenario": g["escenario"].iloc[0], "repeticiones": len(g),
                      "consistencia_pct": round(moda / len(g) * 100, 1)})
    return pd.DataFrame(filas, columns=["consulta_id", "escenario", "repeticiones", "consistencia_pct"])


def consistencia_global(df: pd.DataFrame) -> float:
    c = consistencia_por_consulta(df)
    return round(float(c["consistencia_pct"].mean()), 1) if len(c) else float("nan")


def errores_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["tipo", "turnos"])
    tipos = {
        "error de tool": int((df["errores_tool"] > 0).sum()),
        "sin resultados (RAG)": int((df["sin_resultados"] > 0).sum()),
        "plan inválido": int(df["plan_fallback"].sum()),
        "excepción": int(df["excepcion"].notna().sum()),
        "comportamiento incorrecto": int((df["correcto"] == False).sum()) if "correcto" in df else 0,  # noqa: E712
    }
    return pd.DataFrame({"tipo": list(tipos), "turnos": list(tipos.values())})


def latencia_por_nodo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["nodo", "media_ms", "p95_ms", "participacion_pct"])
    filas = [(n, ms) for d in df["latencia_nodos"] for n, ms in d.items()]
    nodos = pd.DataFrame(filas, columns=["nodo", "ms"])
    g = nodos.groupby("nodo")["ms"].agg(media_ms="mean", p95_ms=lambda s: s.quantile(0.95)).round(1).reset_index()
    g["participacion_pct"] = (g["media_ms"] / g["media_ms"].sum() * 100).round(1)
    return g.sort_values("media_ms", ascending=False)


def latencia_por_tool(df: pd.DataFrame) -> pd.DataFrame:
    filas = [(t["nombre"], t["ms"], t["estado"]) for ts in df.get("tools", []) for t in ts]
    if not filas:
        return pd.DataFrame(columns=["tool", "llamadas", "media_ms", "p95_ms", "fallas"])
    t = pd.DataFrame(filas, columns=["tool", "ms", "estado"])
    g = t.groupby("tool").agg(llamadas=("ms", "count"), media_ms=("ms", "mean"),
                              p95_ms=("ms", lambda s: s.quantile(0.95)),
                              fallas=("estado", lambda s: int((s != "ok").sum()))).round(1).reset_index()
    return g.sort_values("media_ms", ascending=False)


def latencia_por_escenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["escenario", "p50_ms", "p95_ms", "tokens_medios"])
    g = df.groupby("escenario").agg(p50_ms=("latencia_ms", "median"),
                                    p95_ms=("latencia_ms", lambda s: s.quantile(0.95)),
                                    tokens_medios=("tokens_in", "mean")).round(1).reset_index()
    return g.sort_values("p95_ms", ascending=False)

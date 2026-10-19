"""Memoria de largo plazo (por estudiante) en SQLite + recuperación semántica.

Se guarda texto + vector. Persiste entre conversaciones (threads) distintas.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .embeddings import get_embedder


class MemoriaLargoPlazo:
    def __init__(self, db_path: Path = config.DB_APP):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memorias (
                id INTEGER PRIMARY KEY AUTOINCREMENT, usuario_id TEXT NOT NULL,
                texto TEXT NOT NULL, vector BLOB NOT NULL, embedder TEXT NOT NULL,
                creado TEXT NOT NULL)"""
        )
        self.conn.commit()
        self.embedder = get_embedder()

    def _filas(self, usuario_id: str) -> list[tuple[int, str, np.ndarray]]:
        """Devuelve (id, texto, vector); re-embebe filas hechas con otro backend."""
        filas = self.conn.execute(
            "SELECT id, texto, vector, embedder FROM memorias WHERE usuario_id=?", (usuario_id,)
        ).fetchall()
        salida = []
        for id_, texto, blob, emb in filas:
            if emb != self.embedder.name:
                vec = self.embedder.embed([texto])[0]
                self.conn.execute("UPDATE memorias SET vector=?, embedder=? WHERE id=?",
                                  (vec.tobytes(), self.embedder.name, id_))
                self.conn.commit()
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
            salida.append((id_, texto, vec))
        return salida

    def guardar(self, usuario_id: str, texto: str) -> str:
        texto = texto.strip()
        vec = self.embedder.embed([texto])[0]
        for _, existente, v in self._filas(usuario_id):
            if float(v @ vec) > 0.95:  # casi duplicado: no se repite
                return f"Ya estaba registrado: {existente}"
        self.conn.execute(
            "INSERT INTO memorias (usuario_id, texto, vector, embedder, creado) VALUES (?,?,?,?,?)",
            (usuario_id, texto, vec.tobytes(), self.embedder.name, datetime.now().isoformat()),
        )
        self.conn.commit()
        return f"Guardado: {texto}"

    def recordar(self, usuario_id: str, consulta: str, k: int = config.MEMORY_TOP_K) -> list[dict]:
        filas = self._filas(usuario_id)
        if not filas:
            return []
        q = self.embedder.embed([consulta])[0]
        puntuadas = sorted(((float(v @ q), t) for _, t, v in filas), reverse=True)[:k]
        return [{"texto": t, "score": s} for s, t in puntuadas if s >= self.embedder.min_score_memoria]

    def todas(self, usuario_id: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT texto FROM memorias WHERE usuario_id=? ORDER BY id", (usuario_id,))]

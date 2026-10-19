"""Servicios compartidos (singletons perezosos): índice RAG, memoria y base de tickets."""
from __future__ import annotations

import sqlite3

from . import config
from .memoria import MemoriaLargoPlazo
from .rag import RagIndex

_rag: RagIndex | None = None
_memoria: MemoriaLargoPlazo | None = None
_tickets: sqlite3.Connection | None = None


def get_rag() -> RagIndex:
    global _rag
    if _rag is None:
        _rag = RagIndex()
    return _rag


def get_memoria() -> MemoriaLargoPlazo:
    global _memoria
    if _memoria is None:
        _memoria = MemoriaLargoPlazo()
    return _memoria


def get_tickets_db() -> sqlite3.Connection:
    global _tickets
    if _tickets is None:
        _tickets = sqlite3.connect(config.DB_APP, check_same_thread=False)
        _tickets.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, usuario_id TEXT NOT NULL,
                categoria TEXT NOT NULL, prioridad TEXT NOT NULL, resumen TEXT NOT NULL,
                estado TEXT NOT NULL DEFAULT 'abierto', creado TEXT NOT NULL)"""
        )
        _tickets.commit()
    return _tickets


def reset_servicios() -> None:
    """Sólo para tests."""
    global _rag, _memoria, _tickets
    _rag = _memoria = _tickets = None

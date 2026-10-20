"""RAG: indexa los .md de data/docs por secciones y recupera por similitud coseno."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import config
from .embeddings import get_embedder


def _chunks_de_archivo(path: Path) -> list[dict]:
    """Divide un markdown en chunks por encabezado '## '."""
    titulo, seccion, buffer, chunks = path.stem, "Introducción", [], []

    def cerrar():
        texto = " ".join(l.strip() for l in buffer if l.strip())
        if texto:
            # el título de la sección se repite para darle más peso en la búsqueda
            chunks.append({"fuente": path.name, "seccion": seccion,
                           "texto": f"{seccion}. {seccion}. {titulo}: {texto}"})

    for linea in path.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# "):
            titulo = linea[2:].split("(")[0].strip()
        elif linea.startswith("## "):
            cerrar()
            seccion, buffer = linea[3:].strip(), []
        else:
            buffer.append(linea)
    cerrar()
    return chunks


def _hash_docs(docs_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(docs_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class RagIndex:
    def __init__(self, docs_dir: Path = config.DOCS_DIR, index_path: Path = config.RAG_INDEX_PATH):
        self.docs_dir, self.index_path = docs_dir, index_path
        self.chunks: list[dict] = []
        self.vectores: np.ndarray | None = None
        self.embedder = get_embedder()
        self._cargar_o_construir()

    def _cargar_o_construir(self, forzar: bool = False) -> None:
        firma = f"{_hash_docs(self.docs_dir)}|{self.embedder.name}"
        if not forzar and self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if data.get("firma") == firma:
                self.chunks = data["chunks"]
                self.vectores = np.array(data["vectores"], dtype=np.float32)
                return
        self.chunks = [c for p in sorted(self.docs_dir.glob("*.md")) for c in _chunks_de_archivo(p)]
        self.vectores = self.embedder.embed([c["texto"] for c in self.chunks])
        self.index_path.write_text(json.dumps(
            {"firma": firma, "chunks": self.chunks, "vectores": self.vectores.tolist()}),
            encoding="utf-8")

    def reindexar(self) -> int:
        self._cargar_o_construir(forzar=True)
        return len(self.chunks)

    def buscar(self, consulta: str, k: int = config.RAG_TOP_K) -> list[dict]:
        q = self.embedder.embed([consulta])[0]
        scores = self.vectores @ q
        orden = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in orden]

    def buscar_relevantes(self, consulta: str, k: int = config.RAG_TOP_K) -> list[dict]:
        """Sólo resultados sobre el umbral de relevancia del embedder."""
        return [r for r in self.buscar(consulta, k) if r["score"] >= self.embedder.min_score]

"""Seguridad y uso responsable: minimización de datos personales y detección de inyección de prompts."""
from __future__ import annotations

import re
import unicodedata

_PII = [
    ("RUT", re.compile(r"(?<![\w.])\d{1,2}\.?\d{3}\.?\d{3}-[\dkK](?![\w])")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("TARJETA", re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")),
    ("TELEFONO", re.compile(r"(?<!\d)(?:\+?56[\s-]?)?9[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
]

_INYECCION = {
    "ignorar_instrucciones": r"(ignora|olvida|descarta|omite)\b.{0,25}\b(instrucciones|reglas|restricciones)",
    "revelar_prompt": r"(muestra|revela|dime|imprime|repite|cual es)\b.{0,40}\b(prompt|instrucciones del sistema|system prompt)",
    "suplantar_rol": r"(actua|comportate|finge)\b.{0,20}\b(como|que eres)\b.{0,20}\b(administrador|admin|root|desarrollador)",
    "modo_desarrollador": r"(modo desarrollador|developer mode|jailbreak|dan mode)",
    "accion_destructiva": r"(borra|elimina|vacia)\b.{0,20}\b(todos|todas)\b.{0,20}\b(tickets|datos|memorias|registros)",
}


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def redactar_pii(texto: str) -> tuple[str, list[str]]:
    """Reemplaza datos personales por marcas ([RUT], [EMAIL]...). Devuelve (texto, tipos detectados)."""
    tipos: list[str] = []
    for nombre, patron in _PII:
        texto, n = patron.subn(f"[{nombre}]", texto)
        if n:
            tipos.append(nombre)
    return texto, tipos


def detectar_inyeccion(texto: str) -> list[str]:
    """Heurística simple de inyección de prompt. Devuelve los patrones que coinciden."""
    t = _norm(texto)
    return [nombre for nombre, p in _INYECCION.items() if re.search(p, t)]

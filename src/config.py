"""Carga de configuracion y rutas del proyecto.

Centraliza el acceso a config.yaml y a las carpetas de datos, para que el
resto de los modulos no tenga que calcular rutas relativas por su cuenta.
"""
from pathlib import Path

import yaml

# Raiz del proyecto = carpeta que contiene a src/
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Lee config.yaml y lo devuelve como diccionario."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths() -> dict:
    """Devuelve las rutas de trabajo del proyecto."""
    return {
        "root": ROOT,
        "raw": ROOT / "data" / "raw",
        "processed": ROOT / "data" / "processed",
        "experiments": ROOT / "experiments",
    }

#!/usr/bin/env python3
"""
Production-grade I/O utilities for ML + Web app.
Safe, reusable, and deployment-ready.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union, Mapping
import logging

PathLike = Union[str, Path]

# ---------------- LOGGING ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

__all__ = [
    "ensure_outdir",
    "save_json",
    "load_json",
]


# ---------------- DIRECTORY HANDLING ----------------
def ensure_outdir(path: PathLike) -> Path:
    """
    Ensure directory exists (safe for ML pipelines & deployment).
    """
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


# ---------------- SAFE JSON SAVE (ATOMIC WRITE) ----------------
def save_json(obj: Mapping[str, Any] | Any, path: PathLike, *, indent: int = 2) -> Path:
    """
    Save JSON safely using atomic write (prevents file corruption).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    temp_path = p.with_suffix(".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

        temp_path.replace(p)  # atomic move
        logger.info(f"JSON saved: {p}")

        return p.resolve()

    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


# ---------------- SAFE JSON LOAD ----------------
def load_json(path: PathLike) -> Any:
    """
    Load JSON safely with error handling.
    """
    p = Path(path)

    if not p.exists():
        logger.warning(f"JSON file not found: {p}")
        return None

    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to load JSON {path}: {e}")
        raise
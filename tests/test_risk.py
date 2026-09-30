"""Tests del overlay de riesgo recomendado (sizing + entrada escalonada)."""
import pytest

from src.risk import ENTRY_STEP, SIZE_FRAC, overlay_step


def test_overlay_escala_y_limita_el_paso():
    # Desde 0 hacia target 1 (goal escalado = 0.5): primer paso limitado a 1/3.
    assert overlay_step(0.0, 1.0) == pytest.approx(ENTRY_STEP)
    # Con step grande, llega directo al objetivo escalado (0.5).
    assert overlay_step(0.0, 1.0, step=1.0) == pytest.approx(0.5)
    # Cerrar: desde 0.5 hacia 0 tambien va a 1/3.
    assert overlay_step(0.5, 0.0) == pytest.approx(0.5 - ENTRY_STEP)


def test_overlay_llega_al_objetivo_en_pasos():
    pos = 0.0
    for _ in range(10):
        pos = overlay_step(pos, 1.0)
    assert pos == pytest.approx(SIZE_FRAC)

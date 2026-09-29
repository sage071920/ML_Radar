"""plotting.py – Darstellung von Spektrogrammen (unabhängig vom Zieltyp)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # nur für Typangaben; vermeidet doppeltes Laden, wenn spectrogram.py als Skript läuft
    from spectrogram import Spectrogram


def auto_freq_limits(spec: Spectrogram, above_noise_db: float = 20.0,
                     margin: float = 0.3) -> tuple[float, float]:
    """Frequenzbereich [Hz], in dem das Ziel deutlich über dem Rauschen liegt.

    Rauschboden = 10 %-Perzentil der mittleren Leistung pro Frequenz (der Median
    würde bei breitem Micro-Doppler schon im Signal liegen). Gemittelt wird linear,
    nicht in dB: ein dB-Mittel drückt kurze, starke Spitzen (Flashes, Flügelausschläge)
    stark nach unten. Alle Frequenzen, deren
    mittlere Leistung mehr als `above_noise_db` darüber liegt, zählen zum Ziel.
    `margin` erweitert den Bereich um diesen Anteil auf jeder Seite.
    """
    power = 10 * np.log10((10 ** (spec.S_db / 10)).mean(axis=1))
    noise_floor = np.percentile(power, 10)
    active = power > noise_floor + above_noise_db
    if not active.any():
        return spec.f.min(), spec.f.max()

    f_lo, f_hi = spec.f[active].min(), spec.f[active].max()
    pad = max(margin * (f_hi - f_lo), 2 * (spec.f[1] - spec.f[0]))  # min. 2 Frequenzstufen Rand
    return max(f_lo - pad, spec.f.min()), min(f_hi + pad, spec.f.max())


def plot_spectrogram(spec: Spectrogram, ax=None, title: str = "",
                     f_lim: tuple[float, float] | None = None, dyn_range: float = 60.0) -> plt.Axes:
    """Spektrogramm in dB mit zweiter Achse für die Radialgeschwindigkeit.

    spec:      Ergebnis von Radar.spectrogram() (enthält auch die Wellenlänge)
    ax:        vorhandene Achse (z. B. für Subplots); None -> neue Figur
    f_lim:     (f_min, f_max) in Hz; None -> automatisch aus den Daten
    dyn_range: dargestellter Dynamikbereich unterhalb des Maximums [dB]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    mesh = ax.pcolormesh(spec.t, spec.f / 1000, spec.S_db, shading="auto", cmap="viridis",
                         vmin=spec.S_db.max() - dyn_range, vmax=spec.S_db.max())
    ax.set_title(title)
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Dopplerfrequenz [kHz]")

    f_lo, f_hi = f_lim if f_lim is not None else auto_freq_limits(spec)
    ax.set_ylim(f_lo / 1000, f_hi / 1000)

    # Zweite Achse: Frequenz -> Radialgeschwindigkeit (v = f * lambda / 2)
    lam = spec.lam
    sec = ax.secondary_yaxis("right", functions=(lambda fk: fk * 1000 * lam / 2,
                                                 lambda v: v * 2 / lam / 1000))
    sec.set_ylabel("Radialgeschwindigkeit [m/s]")

    ax.figure.colorbar(mesh, ax=ax, label="Echostärke [dB]", pad=0.1)
    return ax

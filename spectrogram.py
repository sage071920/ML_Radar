"""spectrogram.py – Simulation von Radar-Micro-Doppler-Spektrogrammen (Drohne vs. Vogel)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import stft


# ---------------------------------------------------------
# Basisklasse: alles, was das Radar "sehen" kann
# ---------------------------------------------------------
class Target(ABC):
    """Gemeinsame Schnittstelle für alle Ziele."""

    @abstractmethod
    def echo(self, t: np.ndarray, lam: float) -> np.ndarray:
        """Komplexes Echo des Ziels (ohne Rauschen) zu den Zeitpunkten t."""

    @classmethod
    @abstractmethod
    def random(cls, rng: np.random.Generator) -> Target:
        """Erzeugt ein Ziel mit zufälligen, aber realistischen Parametern."""


# ---------------------------------------------------------
# Drohne
# ---------------------------------------------------------
@dataclass
class Drone(Target):
    # Parameter, die man beim Erstellen setzt (alle mit Standardwert)
    v_body: float = 3.0         # Radialgeschwindigkeit [m/s]
    R0: float = 100.0           # Startentfernung [m]
    rpm: float = 6000.0         # Rotordrehzahl [U/min]
    blade_len: float = 0.12     # Blattlänge [m]
    n_rotors: int = 4
    n_blades: int = 2
    n_points: int = 20          # Streupunkte pro Blatt
    elev_deg: float = 20.0      # Blickwinkel auf die Rotorebene [°]
    amp_point: float = 0.03     # Reflexionsstärke pro Blattpunkt
    seed: int = 0               # macht die zufälligen Rotor-Details reproduzierbar

    # Abgeleitete Werte: NICHT im Konstruktor, werden in __post_init__ berechnet
    rotor_speed_factors: np.ndarray = field(init=False, repr=False)
    rotor_phases: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Legt einmalig die zufälligen Abweichungen der einzelnen Rotoren fest."""
        ...

    def echo(self, t: np.ndarray, lam: float) -> np.ndarray:
        """Rumpf + alle Rotorblätter."""
        R_body = self.R0 - self.v_body * t   # Entfernung über die Zeit

        signal = 1.0 * np.exp(-1j * 4 * np.pi * R_body / lam)
        omega = 2 * np.pi * self.rpm / 60
        r = np.linspace(0.01, self.blade_len, self.n_points)
        rng = np.random.default_rng(42)

        cos_elev = np.cos(np.deg2rad(self.elev_deg))

        for k in range(self.n_rotors):
            w = omega * (1 + rng.uniform(-0.03, 0.03))   # Rotoren drehen leicht unterschiedlich
            phi0 = rng.uniform(0, 2 * np.pi)             # zufällige Startstellung
            for b in range(self.n_blades):
                phi = phi0 + b * 2 * np.pi / self.n_blades
                # Radiale Auslenkung jedes Punktes: Kreisbewegung, projiziert auf die Sichtlinie
                # Form: (n_points, Anzahl Samples)
                dR = np.outer(r, np.cos(w * t + phi)) * cos_elev
                R_blade = R_body + dR
                signal += self.amp_point * np.exp(-1j * 4 * np.pi * R_blade / lam).sum(axis=0)

        return signal

    @classmethod
    def random(cls, rng: np.random.Generator) -> Drone:
        ...


# ---------------------------------------------------------
# Vogel
# ---------------------------------------------------------
@dataclass
class Bird(Target):
    v_body: float = 10.0        # Radialgeschwindigkeit [m/s]
    R0: float = 100.0           # Startentfernung [m]
    flap_freq: float = 5.0      # Flügelschläge pro Sekunde [Hz]
    flap_amp_deg: float = 40.0  # maximaler Flügelausschlag [°]
    wing_len: float = 0.3       # Länge eines Flügels [m]
    n_points: int = 20          # Streupunkte pro Flügel
    elev_deg: float = 10.0      # Blickwinkel [°]
    flap_phase: float = 0.0     # Startstellung der Flügel [rad]
    amp_body: float = 1.0
    amp_point: float = 0.05

    def echo(self, t: np.ndarray, lam: float) -> np.ndarray:
        """Körper + zwei schlagende Flügel."""
        ...

    @classmethod
    def random(cls, rng: np.random.Generator) -> Bird:
        ...


# ---------------------------------------------------------
# Ergebnis eines Spektrogramms
# ---------------------------------------------------------
@dataclass(frozen=True)
class Spectrogram:
    f: np.ndarray       # Frequenzachse [Hz]
    t: np.ndarray       # Zeitachse [s]
    S_db: np.ndarray    # Werte [dB], Form (len(f), len(t))

    def plot(self, ax=None, title: str = "") -> None:
        ...


# ---------------------------------------------------------
# Radar: misst Ziele und erstellt Spektrogramme
# ---------------------------------------------------------
@dataclass(frozen=True)
class Radar:
    fc: float = 10e9            # Trägerfrequenz [Hz]
    fs: float = 20_000          # Abtastrate [Hz]
    duration: float = 0.5       # Messdauer [s]
    snr_db: float = 25.0
    nperseg: int = 256          # STFT-Fensterlänge
    noverlap: int = 240

    @property
    def lam(self) -> float:
        """Wellenlänge – wird aus fc berechnet, nicht separat gespeichert."""
        return 3e8 / self.fc

    @property
    def time_axis(self) -> np.ndarray:
        n = int(round(self.duration * self.fs))
        return np.arange(n) / self.fs

    def measure(self, target: Target, rng: np.random.Generator) -> np.ndarray:
        """Echo des Ziels + Rauschen = empfangenes Signal."""
        t = self.time_axis
        signal = target.echo(t, self.lam)
        # ... Rauschen addieren
        p_signal = np.mean(np.abs(signal) ** 2)
        p_noise = p_signal / 10 ** (self.snr_db / 10)
        noise = np.sqrt(p_noise / 2) * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
        return signal + noise

    def spectrogram(self, signal: np.ndarray) -> Spectrogram:
        """STFT, fftshift, Umrechnung in dB."""
        f, t_stft, Z = stft(signal, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, return_onesided=False)
        f = np.fft.fftshift(f)
        Z = np.fft.fftshift(Z, axes=0)
        S_db = 20 * np.log10(np.abs(Z) + 1e-12)

        return Spectrogram(f=f, t=t_stft, S_db=S_db)
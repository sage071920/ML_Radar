from spectrogram import Drone
from spectrogram import Radar
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(0)
radar = Radar(nperseg=64, noverlap=56)

drone = Drone(rpm=7000)
spec = radar.spectrogram(radar.measure(drone, rng))

# ---------------------------------------------------------
# Plot in dB, mit Geschwindigkeitsachse
# ---------------------------------------------------------
lam = radar.lam
omega = 2 * np.pi * drone.rpm / 60
f_tip = 2 * omega * drone.blade_len * np.cos(np.deg2rad(drone.elev_deg)) / lam
f_body = 2 * drone.v_body / lam
print(f"Rumpf-Doppler: {f_body:.0f} Hz, Blattspitzen-Doppler: ±{f_tip:.0f} Hz")

fig, ax = plt.subplots(figsize=(12, 6))
mesh = ax.pcolormesh(spec.t, spec.f / 1000, spec.S_db, shading="auto", cmap="viridis",
                     vmin=spec.S_db.max() - 60, vmax=spec.S_db.max())
ax.set_title(f"Simuliertes Drohnen-Spektrogramm ({drone.rpm:.0f} U/min, SNR {radar.snr_db:.0f} dB)")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Dopplerfrequenz [kHz]")
ax.set_ylim((f_body - 1.3 * f_tip) / 1000, (f_body + 1.3 * f_tip) / 1000)  # Fokus auf Micro-Doppler

# Zweite Achse: Frequenz -> Radialgeschwindigkeit (v = f * lambda / 2)
sec = ax.secondary_yaxis("right", functions=(lambda fk: fk * 1000 * lam / 2,
                                             lambda v: v * 2 / lam / 1000))
sec.set_ylabel("Radialgeschwindigkeit [m/s]")

fig.colorbar(mesh, ax=ax, label="Echostärke [dB]", pad=0.08)
plt.tight_layout()
plt.show()

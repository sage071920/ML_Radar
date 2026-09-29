import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# ---------------------------------------------------------
# 1. Radar-Parameter
# ---------------------------------------------------------
c = 3e8
fc = 10e9                  # X-Band, 10 GHz
lam = c / fc               # Wellenlänge: 3 cm
fs = 20_000                # komplexe Abtastrate -> sichtbarer Bereich: ±10 kHz
duration = 0.5             # Sekunden
t = np.arange(0, duration, 1 / fs)

rng = np.random.default_rng(42)

# ---------------------------------------------------------
# 2. Rumpf: bewegt sich mit konstanter Geschwindigkeit
# ---------------------------------------------------------
R0 = 100.0                 # Startentfernung [m]
v_body = 3.0               # Radialgeschwindigkeit [m/s], positiv = auf das Radar zu
R_body = R0 - v_body * t   # Entfernung über die Zeit

# Das Radar misst die Phase des Echos: Hin- und Rückweg = 2R -> Phase 4*pi*R/lambda
signal = 1.0 * np.exp(-1j * 4 * np.pi * R_body / lam)
# Erwartete Rumpf-Dopplerfrequenz: 2*v/lambda = 200 Hz

# ---------------------------------------------------------
# 3. Rotoren: jedes Blatt = viele Streupunkte auf einem Kreis
# ---------------------------------------------------------
rpm = 7000                 # Zum Ausprobieren: 600 -> einzelne Flashes gut sichtbar
omega = 2 * np.pi * rpm / 60
blade_len = 0.22           # Blattlänge [m]
n_rotors = 4
n_blades = 2               # Blätter pro Rotor
n_points = 20              # Streupunkte pro Blatt
elev = np.deg2rad(20)      # Blickwinkel des Radars auf die Rotorebene
amp_point = 0.03           # Reflexionsstärke pro Blattpunkt (Blätter reflektieren schwach)

r = np.linspace(0.01, blade_len, n_points)  # Abstand der Punkte von der Rotorachse

for k in range(n_rotors):
    w = omega * (1 + rng.uniform(-0.03, 0.03))   # Rotoren drehen leicht unterschiedlich
    phi0 = rng.uniform(0, 2 * np.pi)             # zufällige Startstellung
    for b in range(n_blades):
        phi = phi0 + b * 2 * np.pi / n_blades
        # Radiale Auslenkung jedes Punktes: Kreisbewegung, projiziert auf die Sichtlinie
        # Form: (n_points, Anzahl Samples)
        dR = np.outer(r, np.cos(w * t + phi)) * np.cos(elev)
        R_blade = R_body + dR
        signal += amp_point * np.exp(-1j * 4 * np.pi * R_blade / lam).sum(axis=0)

# Theoretische maximale Micro-Doppler-Frequenz (Blattspitze):
f_tip = 2 * omega * blade_len * np.cos(elev) / lam
print(f"Rumpf-Doppler: {2 * v_body / lam:.0f} Hz, Blattspitzen-Doppler: ±{f_tip:.0f} Hz")

# ---------------------------------------------------------
# 4. Komplexes Rauschen mit einstellbarem SNR
# ---------------------------------------------------------
snr_db = 25                # bezogen auf das Gesamtsignal (Rumpf dominiert!)
p_signal = np.mean(np.abs(signal) ** 2)
p_noise = p_signal / 10 ** (snr_db / 10)
noise = np.sqrt(p_noise / 2) * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
rx = signal + noise

# ---------------------------------------------------------
# 5. STFT: komplexes Signal -> beidseitiges Spektrum
# ---------------------------------------------------------
f, t_stft, Z = stft(rx, fs=fs, nperseg=256, noverlap=240, return_onesided=False)
f = np.fft.fftshift(f)                 # Frequenzen von -fs/2 bis +fs/2 sortieren
Z = np.fft.fftshift(Z, axes=0)
S_db = 20 * np.log10(np.abs(Z) + 1e-12)

# ---------------------------------------------------------
# 6. Plot in dB, mit Geschwindigkeitsachse
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
mesh = ax.pcolormesh(t_stft, f / 1000, S_db, shading="auto", cmap="viridis",
                     vmin=S_db.max() - 60, vmax=S_db.max())
ax.set_title(f"Simuliertes Drohnen-Spektrogramm ({rpm} U/min, SNR {snr_db} dB)")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Dopplerfrequenz [kHz]")
f_body = 2 * v_body / lam
ax.set_ylim((f_body - 1.3 * f_tip) / 1000, (f_body + 1.3 * f_tip) / 1000)  # Fokus auf Micro-Doppler

# Zweite Achse: Frequenz -> Radialgeschwindigkeit (v = f * lambda / 2)
sec = ax.secondary_yaxis("right", functions=(lambda fk: fk * 1000 * lam / 2,
                                             lambda v: v * 2 / lam / 1000))
sec.set_ylabel("Radialgeschwindigkeit [m/s]")

fig.colorbar(mesh, ax=ax, label="Echostärke [dB]", pad=0.08)
plt.tight_layout()
plt.show()
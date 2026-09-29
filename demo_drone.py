from spectrogram import Drone
from spectrogram import Radar
from spectrogram import Bird
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(0)
radar = Radar(duration=4.0, nperseg=64, noverlap=56)
# Vogel: fs=2000 reicht (Doppler < 1 kHz), 0,6-s-Fenster für feine Frequenzauflösung,
# nfft=8192 (Zero-Padding) für ein glattes Frequenzraster
bird_radar = Radar(duration=30.0, fs=2000, nperseg=1200, noverlap=1180, nfft=8192)


drone = Drone(rpm=7000)
bird = Bird(flap_freq=0.75)
drone_spec = radar.spectrogram(radar.measure(drone, rng))
bird_spec = bird_radar.spectrogram(bird_radar.measure(bird, rng))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
drone_spec.plot(ax=ax1, title=f"Drohne ({drone.rpm:.0f} U/min, SNR {radar.snr_db:.0f} dB)")
bird_spec.plot(ax=ax2, title=f"Vogel ({bird.flap_freq:g} Hz Flügelschlag, SNR {bird_radar.snr_db:.0f} dB)")
plt.tight_layout()
plt.show()

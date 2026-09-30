import numpy as np

from spectrogram import Bird, Drone, Radar


def generate_dataset(radar, n_per_class, rng):
    X, y = [], []
    for label, cls in enumerate([Bird, Drone]):   # 0 = Vogel, 1 = Drohne
        for _ in range(n_per_class):
            target = cls.random(rng)
            spec = radar.spectrogram(radar.measure(target, rng))
            X.append(spec.S_db - spec.S_db.max())
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y), spec.f, spec.t



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from spectrogram import Spectrogram

    radar = Radar(duration=1.0, nperseg=256, noverlap=192)
    X, y, f, t = generate_dataset(radar, n_per_class=5, rng=np.random.default_rng(4))
    print(X.shape, f"{X.nbytes / 1e6:.1f} MB")

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, S, label in zip(axes.flat, X, y):
        Spectrogram(f=f, t=t, S_db=S, lam=radar.lam).plot(
            ax=ax, title="Drohne" if label else "Vogel", f_lim=(-10_000, 10_000))
    plt.tight_layout()
    plt.show()
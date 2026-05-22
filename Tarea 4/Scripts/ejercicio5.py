import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# Parámetros
fs = 1000
T = 2.0
t = np.linspace(0, T, int(fs * T), endpoint=False)
N = len(t)

# Señal A (Chirp): s1(t) = sin(2*pi*(10 + 40t)*t)
s1 = np.sin(2 * np.pi * (10 + 40 * t) * t)

# Señal B (Salto de Frecuencia): s2(t) = sin(2*pi*20*t) para t < 1, y sin(2*pi*80*t) para t >= 1
s2 = np.piecewise(t, [t < 1, t >= 1], [
    lambda t: np.sin(2 * np.pi * 20 * t), 
    lambda t: np.sin(2 * np.pi * 80 * t)
])

# Transformada de Fourier
S1 = fft(s1)
S2 = fft(s2)
freqs = fftfreq(N, 1/fs)

# Filtro Pasa-Altas
cutoff = 50
high_pass_filter = np.abs(freqs) >= cutoff

# Aplica el filtro en el dominio de la frecuencia
S1_filtered = S1 * high_pass_filter
S2_filtered = S2 * high_pass_filter

# Aplica la Transformada Inversa para obtener señales filtradas en el tiempo
s1_filtered = np.real(ifft(S1_filtered))
s2_filtered = np.real(ifft(S2_filtered))

# Graficar
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# Señales Originales en el Tiempo
axs[0, 0].plot(t, s1)
axs[0, 0].set_title('Señal A (Chirp) en el Tiempo')
axs[0, 0].set_xlabel('Tiempo [s]')
axs[0, 0].set_ylabel('Amplitud')
axs[0, 0].grid(True)

axs[0, 1].plot(t, s2)
axs[0, 1].set_title('Señal B (Salto de Frecuencia) en el Tiempo')
axs[0, 1].set_xlabel('Tiempo [s]')
axs[0, 1].set_ylabel('Amplitud')
axs[0, 1].grid(True)

# Espectros de Frecuencia (Solo frecuencias positivas)
pos_freqs = freqs[:N//2]
mag_S1 = np.abs(S1[:N//2]) * 2 / N
mag_S2 = np.abs(S2[:N//2]) * 2 / N

axs[1, 0].plot(pos_freqs, mag_S1)
axs[1, 0].set_title('Espectro Señal A')
axs[1, 0].set_xlabel('Frecuencia [Hz]')
axs[1, 0].set_ylabel('Magnitud')
axs[1, 0].set_xlim(0, 120)
axs[1, 0].grid(True)

axs[1, 1].plot(pos_freqs, mag_S2)
axs[1, 1].set_title('Espectro Señal B')
axs[1, 1].set_xlabel('Frecuencia [Hz]')
axs[1, 1].set_ylabel('Magnitud')
axs[1, 1].set_xlim(0, 120)
axs[1, 1].grid(True)

# Señales Filtradas en el Tiempo
axs[2, 0].plot(t, s1_filtered, color='orange')
axs[2, 0].set_title('Señal A Filtrada (Pasa-Altas > 50Hz)')
axs[2, 0].set_xlabel('Tiempo [s]')
axs[2, 0].set_ylabel('Amplitud')
axs[2, 0].grid(True)

axs[2, 1].plot(t, s2_filtered, color='orange')
axs[2, 1].set_title('Señal B Filtrada (Pasa-Altas > 50Hz)')
axs[2, 1].set_xlabel('Tiempo [s]')
axs[2, 1].set_ylabel('Amplitud')
axs[2, 1].grid(True)

plt.tight_layout()
plt.savefig('/home/kryptonita/Documentos/Analisis-Matem-tico-Aplicado/Tarea 4/LaTeX/ejercicio5_grafica.png')
plt.close()

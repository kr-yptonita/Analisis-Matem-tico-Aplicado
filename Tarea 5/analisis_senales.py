import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.datasets import electrocardiogram
import os

os.makedirs('figuras', exist_ok=True)
plt.rcParams['figure.figsize'] = (10, 5)

# carga de electrocardiograma 
fs = 360.0
N_muestras = 5 * int(fs)
ecg_completo = electrocardiogram()
t = np.arange(N_muestras) / fs
senial = ecg_completo[:N_muestras]

# ej 4: cwt y escalograma
plt.figure()
plt.plot(t, senial, color='navy')
plt.title('Señal Original (ECG)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.grid(True)
plt.tight_layout()
plt.savefig('figuras/ej4_senal_original.png')
plt.close()

escalas = np.arange(1, 128)
coeficientes, frecuencias = pywt.cwt(senial, escalas, 'cmor1.5-1.0', sampling_period=1/fs)

plt.figure(figsize=(10, 6))
magnitud = np.abs(coeficientes)
plt.imshow(magnitud, extent=[0, t[-1], frecuencias[-1], frecuencias[0]], cmap='jet', aspect='auto',
           vmax=abs(coeficientes).max(), vmin=-abs(coeficientes).max())
plt.title('Escalograma CWT (Wavelet Morlet Compleja)')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(label='Magnitud')
plt.tight_layout()
plt.savefig('figuras/ej4_escalograma.png')
plt.close()

# ej 5: quitar el ruido usando dwt
np.random.seed(42)
ruido = np.random.normal(0, 0.5, len(senial))
senial_ruidosa = senial + ruido

wavelet = 'db4'
nivel = 4
coefs = pywt.wavedec(senial_ruidosa, wavelet, level=nivel)
cA4, cD4, cD3, cD2, cD1 = coefs

fig, axs = plt.subplots(5, 1, figsize=(10, 10))
axs[0].plot(cA4); axs[0].set_title('Aproximación A4')
axs[1].plot(cD4); axs[1].set_title('Detalle D4')
axs[2].plot(cD3); axs[2].set_title('Detalle D3')
axs[3].plot(cD2); axs[3].set_title('Detalle D2')
axs[4].plot(cD1); axs[4].set_title('Detalle D1')
for ax in axs: ax.grid(True)
plt.tight_layout()
plt.savefig('figuras/ej5_coeficientes.png')
plt.close()

mad = np.median(np.abs(cD1 - np.median(cD1)))
sigma = mad / 0.6745
umbral_lambda = sigma * np.sqrt(2 * np.log(len(senial_ruidosa)))

coefs_filtrados = [cA4]
for i in range(1, len(coefs)):
    coefs_filtrados.append(pywt.threshold(coefs[i], value=umbral_lambda, mode='soft'))

senial_filtrada = pywt.waverec(coefs_filtrados, wavelet)
senial_filtrada = senial_filtrada[:len(senial_ruidosa)]

plt.figure(figsize=(12, 6))
plt.plot(t, senial_ruidosa, color='lightgray', label='Señal Ruidosa', alpha=0.7)
plt.plot(t, senial_filtrada, color='red', label='Señal Filtrada (Wavelet Shrinkage)', linewidth=1.5)
plt.plot(t, senial - 4, color='blue', label='Señal Original (desplazada -4)', linewidth=1)
plt.title('Denoising mediante Análisis Multirresolución')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figuras/ej5_denoising.png')
plt.close()

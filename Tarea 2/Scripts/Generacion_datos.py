import numpy as np

def f(x, m):
    a, b, c = m
    return a + np.sin(2 * np.pi * b * x) / (x ** c)

def generar_datos(N=200, m_real=[2.0, 1.5, 0.5], sigma=0.5, semilla=42):
    np.random.seed(semilla)
    x = np.linspace(0.5, 10, N)
    y_limpio = f(x, m_real)
    ruido = np.random.normal(0, sigma, N)
    y = y_limpio + ruido
    return x, y_limpio, y, m_real

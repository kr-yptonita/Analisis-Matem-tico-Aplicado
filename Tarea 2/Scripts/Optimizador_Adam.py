import numpy as np
import matplotlib.pyplot as plt
import os
from Generacion_datos import f, generar_datos

def gradiente_f(x, y, m):
    a, b, c = m
    error = f(x, m) - y
    
    df_da = np.ones_like(x)
    df_db = (2 * np.pi * x * np.cos(2 * np.pi * b * x)) / (x ** c)
    df_dc = - (np.sin(2 * np.pi * b * x) * np.log(x)) / (x ** c)
    
    grad_a = np.mean(error * df_da)
    grad_b = np.mean(error * df_db)
    grad_c = np.mean(error * df_dc)
    
    return np.array([grad_a, grad_b, grad_c])

def optimizador_adam(x, y, m0, alfa=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, epocas=1000):
    m = np.array(m0, dtype=float)
    
    M = np.zeros_like(m)
    V = np.zeros_like(m)
    
    historial_perdida = []
    
    for t in range(1, epocas + 1):
        gt = gradiente_f(x, y, m)
        
        perdida = 0.5 * np.mean((f(x, m) - y)**2)
        historial_perdida.append(perdida)
        
        M = beta1 * M + (1 - beta1) * gt
        V = beta2 * V + (1 - beta2) * (gt ** 2)
        
        M_hat = M / (1 - beta1 ** t)
        V_hat = V / (1 - beta2 ** t)
        
        m = m - alfa * M_hat / (np.sqrt(V_hat) + epsilon)
        
    return m, historial_perdida

if __name__ == "__main__":
    # Generar datos
    x, y_limpio, y, m_real = generar_datos(sigma=0.5)
    
    # Inicialización errónea como pide el inciso D
    m0 = [0.0, 0.5, 1.0]
    
    # Directorio de salida para las figuras
    out_dir = '../LaTeX'
    
    # --- Inciso D.1 y D.2 ---
    m_final, historial_perdida = optimizador_adam(x, y, m0, alfa=0.05, epocas=2000)
    print(f"Parámetros reales: {m_real}")
    print(f"Parámetros finales (alfa=0.05): {np.round(m_final, 4)}")
    
    # Gráfica de convergencia (Loss vs Epochs)
    plt.figure(figsize=(8, 5))
    plt.plot(historial_perdida, color='teal', linewidth=2)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.title('Convergencia del Optimizador Adam ($\\alpha=0.05$)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_loss_005.png'))
    
    # Gráfica de ajuste de curva
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Datos ruidosos', color='lightgreen', alpha=0.6, s=15)
    plt.plot(x, y_limpio, label=f'Modelo real $m_{{real}}={m_real}$', color='navy', linestyle='--', linewidth=2)
    plt.plot(x, f(x, m_final), label=f'Modelo ajustado $m_{{final}}={list(np.round(m_final, 3))}$', color='crimson', linestyle='-', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajuste de Curva con Optimizador Adam')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_ajuste.png'))
    
    # --- Inciso D.3: Impacto de la Tasa de Aprendizaje ---
    m_final_lento, historial_perdida_lento = optimizador_adam(x, y, m0, alfa=0.001, epocas=2000)
    print(f"Parámetros finales (alfa=0.001): {np.round(m_final_lento, 4)}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(historial_perdida, label='$\\alpha = 0.05$', color='teal', linewidth=2)
    plt.plot(historial_perdida_lento, label='$\\alpha = 0.001$', color='coral', linewidth=2)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.title('Comparación de Tasas de Aprendizaje')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_alpha_comp.png'))

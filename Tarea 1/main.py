import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from optimizador import (
    modelo,
    inversa_generalizada,
    descenso_gradiente,
    gauss_newton,
    levenberg_marquardt
)

# Parámetros reals para la generación de datos sintéticos
P_VERDADERO = [2.0, 0.5, 1.5] # a, b, c

class AplicacionAjusteCurvas:
    def __init__(self, root):
        self.root = root
        self.root.title("Ajuste de Curvas con Algoritmos de Gradiente")
        self.root.geometry("1200x800")
        
        # Generar datos
        self.generar_datos()
        self.p_inicial = [1.0, 0.1, 1.0] # Suposición inicial
        
        # Cargar imágenes
        self.cargar_imagenes()
        
        # Configurar UI
        self.configurar_ui()
        
    def generar_datos(self):
        np.random.seed(42)  # Para reproducibilidad
        self.x_datos = np.linspace(0.1, 5.0, 50)
        # y = f(x) + ruido
        ruido = np.random.normal(0, 0.5, size=self.x_datos.shape)
        self.y_datos = modelo(self.x_datos, P_VERDADERO) + ruido
        
    def cargar_imagenes(self):
        ruta_base = os.path.dirname(os.path.abspath(__file__))
        ruta_fondo = os.path.join(ruta_base, "Fondo.png")
        
        
        ruta_logo = os.path.join(ruta_base, "Logo para GUI.png")
        if not os.path.exists(ruta_logo):
            ruta_logo_alt = os.path.join(ruta_base, "Logo para la GUI.png")
            if os.path.exists(ruta_logo_alt):
                ruta_logo = ruta_logo_alt
        
        try:
            # Fondo
            img_fondo = Image.open(ruta_fondo)
            img_fondo = img_fondo.resize((1200, 800), Image.Resampling.LANCZOS)
            self.img_fondo_tk = ImageTk.PhotoImage(img_fondo)
            
            
            img_logo = Image.open(ruta_logo)
            img_logo = img_logo.resize((150, 150), Image.Resampling.LANCZOS)
            self.img_logo_tk = ImageTk.PhotoImage(img_logo)
            
        except Exception as e:
            print(f"Error al cargar las imágenes: {e}")
            self.img_fondo_tk = None
            self.img_logo_tk = None

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionAjusteCurvas(root)
    root.mainloop()
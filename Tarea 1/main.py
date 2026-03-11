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

    def configurar_ui(self):
        # 1. Canvas para el fondo
        self.fondo_canvas = tk.Canvas(self.root, width=1200, height=800)
        self.fondo_canvas.pack(fill="both", expand=True)
        
        if self.img_fondo_tk:
            self.fondo_canvas.create_image(0, 0, image=self.img_fondo_tk, anchor="nw")
            
        if self.img_logo_tk:
            self.fondo_canvas.create_image(50, 50, image=self.img_logo_tk, anchor="nw")
            
        # 2. Panel de Controles
        marco_controles = tk.Frame(self.fondo_canvas, bg="lightgray", bd=2, relief="groove")
        self.fondo_canvas.create_window(50, 220, window=marco_controles, anchor="nw")
        
        tk.Label(marco_controles, text="Opciones de Ajuste", font=("Arial", 14, "bold"), bg="lightgray").pack(pady=10)
        
        tk.Label(marco_controles, text="Seleccione el Algoritmo:", bg="lightgray").pack(anchor="w", padx=10)
        
        self.algoritmo_var = tk.StringVar(value="Levenberg-Marquardt")
        algoritmos = [
            "Inversa Generalizada", 
            "Descenso del Gradiente", 
            "Gauss-Newton", 
            "Levenberg-Marquardt"
        ]
        
        for alg in algoritmos:
            tk.Radiobutton(marco_controles, text=alg, variable=self.algoritmo_var, value=alg, bg="lightgray").pack(anchor="w", padx=20)
            
        btn_ajustar = tk.Button(marco_controles, text="Ajustar Curva", command=self.ejecutar_ajuste, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
        btn_ajustar.pack(pady=20, padx=10, fill="x")
        
        # Mostrar parámetros iniciales
        tk.Label(marco_controles, text="Parámetros Reales:", font=("Arial", 10, "bold"), bg="lightgray").pack(anchor="w", padx=10, pady=(10,0))
        tk.Label(marco_controles, text=f"a={P_VERDADERO[0]}, b={P_VERDADERO[1]}, c={P_VERDADERO[2]}", bg="lightgray").pack(anchor="w", padx=20)
        
        tk.Label(marco_controles, text="Estimación Inicial:", font=("Arial", 10, "bold"), bg="lightgray").pack(anchor="w", padx=10, pady=(10,0))
        tk.Label(marco_controles, text=f"a={self.p_inicial[0]}, b={self.p_inicial[1]}, c={self.p_inicial[2]}", bg="lightgray").pack(anchor="w", padx=20)
        
        # Resultados
        self.lbl_resultados = tk.Label(marco_controles, text="", bg="lightgray", justify="left")
        self.lbl_resultados.pack(pady=10, padx=10)
        
        # 3. Panel de Gráficas (Matplotlib) incorporado en la ventana
        # Colocamos un frame derecho para las dos gráficas
        marco_graficas = tk.Frame(self.fondo_canvas, bg="white", bd=2, relief="sunken")
        self.fondo_canvas.create_window(300, 50, window=marco_graficas, anchor="nw", width=850, height=700)
        
        self.figura = Figure(figsize=(10, 8), dpi=100)
        self.figura.subplots_adjust(hspace=0.3)
        self.ax_ajuste = self.figura.add_subplot(211)
        self.ax_convergencia = self.figura.add_subplot(212)
        
        self.canvas_plot = FigureCanvasTkAgg(self.figura, master=marco_graficas)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.actualizar_grafica_inicial()

    def actualizar_grafica_inicial(self):
        self.ax_ajuste.clear()
        self.ax_ajuste.plot(self.x_datos, self.y_datos, 'o', label='Datos Observados (Con Ruido)')
        
        x_continuo = np.linspace(min(self.x_datos), max(self.x_datos), 200)
        y_inicial = modelo(x_continuo, self.p_inicial)
        self.ax_ajuste.plot(x_continuo, y_inicial, 'r--', label='Suposición Inicial')
        
        self.ax_ajuste.set_title("Ajuste de Datos")
        self.ax_ajuste.set_xlabel("x")
        self.ax_ajuste.set_ylabel("f(x)")
        self.ax_ajuste.legend()
        self.ax_ajuste.grid(True)
        
        self.ax_convergencia.clear()
        self.ax_convergencia.set_title("Convergencia del Costo F(p)")
        self.ax_convergencia.set_xlabel("Iteración")
        self.ax_convergencia.set_ylabel("Costo (Log)")
        self.ax_convergencia.grid(True)
        
        self.canvas_plot.draw()

    def ejecutar_ajuste(self):
        algSeleccionado = self.algoritmo_var.get()
        p_optimos = None
        historial = None
        
        try:
            if algSeleccionado == "Inversa Generalizada":
                p_optimos, historial = inversa_generalizada(self.x_datos, self.y_datos, self.p_inicial, iteraciones=50)
            elif algSeleccionado == "Descenso del Gradiente":
                # Tasa de aprendizaje muy pequeña porque la función puede crecer rápidamente
                p_optimos, historial = descenso_gradiente(self.x_datos, self.y_datos, self.p_inicial, tasa_aprendizaje=0.0001, iteraciones=5000)
            elif algSeleccionado == "Gauss-Newton":
                p_optimos, historial = gauss_newton(self.x_datos, self.y_datos, self.p_inicial, iteraciones=100)
            elif algSeleccionado == "Levenberg-Marquardt":
                p_optimos, historial = levenberg_marquardt(self.x_datos, self.y_datos, self.p_inicial, iteraciones=100)
                
            self.mostrar_resultados(p_optimos, historial)
            self.graficar_resultados(p_optimos, historial, algSeleccionado)
            
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"El algoritmo falló o no convergió.\n{str(e)}")

    def mostrar_resultados(self, p_optimos, historial):
        texto = f"Parámetros Optimizados:\n"
        texto += f"a = {p_optimos[0]:.4f}\n"
        texto += f"b = {p_optimos[1]:.4f}\n"
        texto += f"c = {p_optimos[2]:.4f}\n\n"
        texto += f"Iteraciones: {len(historial)}\n"
        texto += f"Error Final: {historial[-1]:.4f}"
        self.lbl_resultados.config(text=texto)

    def graficar_resultados(self, p_optimos, historial, algoritmo):
        self.ax_ajuste.clear()
        self.ax_ajuste.plot(self.x_datos, self.y_datos, 'o', label='Datos Observados')
        
        x_continuo = np.linspace(min(self.x_datos), max(self.x_datos), 500)
        y_ajustada = modelo(x_continuo, p_optimos)
        
        self.ax_ajuste.plot(x_continuo, y_ajustada, 'g-', linewidth=2, label=f'Curva Ajustada ({algoritmo})')
        self.ax_ajuste.set_title(f"Ajuste de Datos - {algoritmo}")
        self.ax_ajuste.set_xlabel("x")
        self.ax_ajuste.set_ylabel("f(x)")
        self.ax_ajuste.legend()
        self.ax_ajuste.grid(True)
        
        self.ax_convergencia.clear()
        self.ax_convergencia.plot(range(1, len(historial)+1), historial, 'b-', linewidth=2)
        self.ax_convergencia.set_title(f"Convergencia del Costo F(p)")
        self.ax_convergencia.set_xlabel("Iteración")
        self.ax_convergencia.set_ylabel("Costo (Escala Log)")
        self.ax_convergencia.set_yscale('log')
        self.ax_convergencia.grid(True)
        
        self.canvas_plot.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionAjusteCurvas(root)
    root.mainloop()
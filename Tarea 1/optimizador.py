import numpy as np

def modelo(x, parametros):
    """
    Evalúa el modelo matemático: f(x) = a + sin(2*pi*b*x) + x^c
    donde parametros = [a, b, c]
    a: Fase
    b: Frecuencia
    c: Modulación y Amplitud
    """
    a, b, c = parametros
    # Se usa np.abs(x) o se asegura x > 0 para evitar errores con x^c si x < 0 y c es fraccionario
    # En este contexto, asumiremos que x > 0 o que los datos lo permiten.
    # Para ser robustos usando base np.clip para que no sea exactamente 0 al elevar a potencia <=0
    x_seguro = np.maximum(x, 1e-8) 
    return a + np.sin(2 * np.pi * b * x) + np.power(x_seguro, c)

def jacobiano(x, parametros):
    """
    Calcula la matriz Jacobiana del modelo con respecto a los parámetros [a, b, c].
    """
    a, b, c = parametros
    x_seguro = np.maximum(x, 1e-8)
    
    n_datos = len(x)
    J = np.zeros((n_datos, 3))
    
    # df/da = 1
    J[:, 0] = 1.0
    
    # df/db = 2 * pi * x * cos(2 * pi * b * x)
    J[:, 1] = 2 * np.pi * x * np.cos(2 * np.pi * b * x)
    
    # df/dc = x^c * ln(x)
    J[:, 2] = np.power(x_seguro, c) * np.log(x_seguro)
    
    return J

def error_residual(x, y, parametros):
    """
    Calcula el vector de residuos: r = F(x, p) - y
    """
    return modelo(x, parametros) - y

def funcion_costo(x, y, parametros):
    """
    Suma de los residuos al cuadrado.
    """
    residuos = error_residual(x, y, parametros)
    return 0.5 * np.sum(residuos**2)
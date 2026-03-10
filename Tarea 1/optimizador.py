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

def inversa_generalizada(x, y, p_inicial, iteraciones=50, tol=1e-6):
    """
    Pseudoinversa de Moore-Penrose.
    Iterativamente actualiza los parámetros usando: p = p - J^+ r
    """
    p = np.array(p_inicial, dtype=float)
    historial_costo = []
    
    for i in range(iteraciones):
        r = error_residual(x, y, p)
        costo = funcion_costo(x, y, p)
        historial_costo.append(costo)
        
        J = jacobiano(x, p)
        J_pinv = np.linalg.pinv(J)
        
        delta_p = J_pinv @ r
        p = p - delta_p
        
        if np.linalg.norm(delta_p) < tol:
            break
            
    return p, historial_costo

def descenso_gradiente(x, y, p_inicial, tasa_aprendizaje=0.001, iteraciones=1000, tol=1e-6):
    """
    Descenso del Gradiente
    """
    p = np.array(p_inicial, dtype=float)
    historial_costo = []
    
    for i in range(iteraciones):
        r = error_residual(x, y, p)
        costo = funcion_costo(x, y, p)
        historial_costo.append(costo)
        
        J = jacobiano(x, p)
        gradiente = J.T @ r
        
        p = p - tasa_aprendizaje * gradiente
        
        if np.linalg.norm(gradiente) < tol:
            break
            
    return p, historial_costo

def gauss_newton(x, y, p_inicial, iteraciones=50, tol=1e-6):
    """
    Algoritmo de Gauss-Newton
    """
    p = np.array(p_inicial, dtype=float)
    historial_costo = []
    
    for i in range(iteraciones):
        r = error_residual(x, y, p)
        costo = funcion_costo(x, y, p)
        historial_costo.append(costo)
        
        J = jacobiano(x, p)
        
        try:
            # (J^T J)^{-1} J^T r
            delta_p = np.linalg.solve(J.T @ J, J.T @ r)
        except np.linalg.LinAlgError:
            # Si la matriz es singular, usamos pseudoinversa como respaldo
            delta_p = np.linalg.pinv(J.T @ J) @ (J.T @ r)
            
        p = p - delta_p
        
        if np.linalg.norm(delta_p) < tol:
            break
            
    return p, historial_costo

def levenberg_marquardt(x, y, p_inicial, lambda_inicial=0.01, factor=10.0, iteraciones=100, tol=1e-6):
    """
    Algoritmo de Levenberg-Marquardt
    """
    p = np.array(p_inicial, dtype=float)
    lambda_val = lambda_inicial
    historial_costo = []
    
    for i in range(iteraciones):
        r = error_residual(x, y, p)
        costo = funcion_costo(x, y, p)
        historial_costo.append(costo)
        
        J = jacobiano(x, p)
        I = np.eye(len(p))
        
        # (J^T J + lambda * I) * delta_p = J^T r
        H = J.T @ J
        # Puede usarse la diagonal de H en lugar de I según algunas variantes
        matriz_lm = H + lambda_val * np.diag(np.diag(H)) 
        
        try:
            delta_p = np.linalg.solve(matriz_lm, J.T @ r)
        except np.linalg.LinAlgError:
            delta_p = np.linalg.pinv(matriz_lm) @ (J.T @ r)
            
        p_nuevo = p - delta_p
        costo_nuevo = funcion_costo(x, y, p_nuevo)
        
        if costo_nuevo < costo:
            # Iteración exitosa
            p = p_nuevo
            lambda_val /= factor
            if np.linalg.norm(delta_p) < tol:
                break
        else:
            # Iteración fallida, aumenta amortiguamiento
            lambda_val *= factor
            # No actualizamos 'p'
            
    return p, historial_costo
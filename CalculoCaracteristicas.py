import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.stats import skew

matriz_filament = [] # Reemplazar dependiendo el archivo a leer
num_puntos_list = [5000, 10000, 15000, 25000]

# Función para realizar los cálculos de características geométricas de una nube de puntos
def realizar_calculos(X, Y, Z):
    puntos = np.column_stack((X, Y, Z))
    
    # CONCAVIDAD (diferencia entre volumen de la nube y volumen de la envolvente convexa)
    if len(puntos) >= 4:  # ConvexHull requiere al menos 4 puntos
        hull = ConvexHull(puntos)
        volumen_hull = hull.volume
        volumen_nube = np.abs(np.linalg.det(np.cov(puntos.T)))  # Aproximación de volumen
        concavidad = volumen_hull - volumen_nube
    else:
        concavidad = 0 
        volumen_hull = 0
        volumen_nube = 0

    # CURVATURA (cambio de la tangente a la curva en una superficie)
    if len(puntos) >= 3:
        curvatura = np.mean(np.abs(np.gradient(np.gradient(Z))))
    else:
        curvatura = 0

    # DENSIDAD DE LAS NORMALES (variación en las direcciones de las normales)
    if len(puntos) >= 3:
        grad_x = np.gradient(X)
        grad_y = np.gradient(Y)
        grad_z = np.gradient(Z)
        normales = np.column_stack((grad_x, grad_y, grad_z))
        densidad_normales = np.std(np.linalg.norm(normales, axis=1))
    else:
        densidad_normales = 0

    # VOLUMEN DE LA ENVOLVENTE CONVEXA
    volumen_convexa = volumen_hull

    # ÁREA DE SUPERFICIE TOTAL
    area_superficie_total = hull.area if len(puntos) >= 4 else 0

    # RADIO DE CURVATURA MEDIO
    radio_curvatura_medio = np.mean(np.sqrt(1 + (np.gradient(Z)**2))) if len(Z) > 1 else 0

    # DESVIACIÓN ESTÁNDAR DE LA ALTURA (coordenadas Z)
    desviacion_std_altura = np.std(Z)

    # COMPACTICIDAD (relación entre volumen y área de superficie)
    compacticidad = volumen_convexa / area_superficie_total if area_superficie_total != 0 else 0

    # DESVIACIÓN ESTÁNDAR DE LAS DISTANCIAS AL CENTROIDE
    centroide = np.mean(puntos, axis=0)
    distancias_centroide = np.linalg.norm(puntos - centroide, axis=1)
    desviacion_std_centroide = np.std(distancias_centroide)

    # Skewness de la distribución de coordenadas
    skewness_x = skew(X)
    skewness_y = skew(Y)
    skewness_z = skew(Z)
    asimetria_promedio = (skewness_x + skewness_y + skewness_z) / 3

    return (concavidad, curvatura, densidad_normales, volumen_convexa,
            area_superficie_total, radio_curvatura_medio, desviacion_std_altura,
            compacticidad, desviacion_std_centroide, asimetria_promedio)

# Iterar sobre los archivos
for i in range(1, 201):
    archivo = f'Filament\Filament_{i}.csv'
    datos = pd.read_csv(archivo)

    X, Y, Z = datos['X'], datos['Y'], datos['Z']

    # Verificar que la nube de puntos tiene suficientes puntos
    if len(datos) >= max(num_puntos_list):
        for num_puntos in num_puntos_list:
            indices_aleatorios = np.random.choice(len(datos), num_puntos, replace=False)
            # Extraer los puntos seleccionados y realizarlos calculos
            X_sel, Y_sel, Z_sel = X[indices_aleatorios], Y[indices_aleatorios], Z[indices_aleatorios]
            caracteristicas = realizar_calculos(X_sel, Y_sel, Z_sel)
            matriz_filament.append(caracteristicas)
    else:
        raise ValueError('La nube de puntos no contiene suficientes puntos para la extracción.')

# Crear DataFrame con nombres de columnas
columnas = ['Concavidad', 'Curvatura', 'DensidadNormales', 'VolumenConvexa', 
            'AreaSuperficieTotal', 'RadioCurvaturaMedio', 'DesviacionStdAltura', 
            'Compacticidad', 'DesviacionStdCentroide', 'AsimetriaPromedio']

tabla_filament = pd.DataFrame(matriz_filament, columns=columnas)
tabla_filament.to_csv('Resultados_filament.csv', index=False)
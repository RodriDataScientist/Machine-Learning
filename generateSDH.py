import numpy as np
import os
import pandas as pd
import featureExtraction as fe
import plotHistograms as ph

# Función para calcular las características de un archivo CSV
def process_csv_file(file_title, col_titles, plotHist=False, output_csv="dataset.csv"):
    db = pd.read_csv(file_title)
    
    # Inicializar una lista para almacenar todas las características
    all_characteristics = []

    # Función para calcular la imagen de sumas y diferencias
    def generate_sum_diff(signal, window_size):
        sum_image = []
        diff_image = []
        
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            sum_image.append(np.sum(window))  # Suma de la ventana
            diff_image.append(np.diff(window).sum())  # Diferencia de la ventana
        
        return np.array(sum_image), np.array(diff_image)

    # Ciclo para recorrer las columnas Col1, Col2, Col3
    for col_title in col_titles:
        p = db[col_title]
        
        # Ciclo para recorrer los tamaños de ventana impares de 3 a 25
        for window_size in range(3, 26, 2):
            # Generar imágenes de suma y diferencia para la señal
            sum, diff = generate_sum_diff(p, window_size)

            # 36 bins
            sum_bins = 36
            diff_bins = 36

            # Crear los histogramas de las imágenes de suma y diferencia con los bins ajustados
            hist_sum, bin_edges_sum = np.histogram(sum, bins=sum_bins, density=True)
            hist_diff, bin_edges_diff = np.histogram(diff, bins=diff_bins, density=True)
            if plotHist:
                ph.plot_histograms(hist_sum, hist_diff, bin_edges_sum, bin_edges_diff, f'Señal {col_title} - Window Size {window_size}')

            # Calcular características para la imagen de suma y diferencia usando los bin_edges
            mean_sum = fe.mean(hist_sum, sum)
            variance_sum = fe.variance(hist_sum, hist_diff, mean_sum, sum, diff)
            correlation_sum = fe.correlation(hist_sum, hist_diff, mean_sum, sum, diff)
            contrast_sum = fe.contrast(hist_diff, diff)
            homogeneity_sum = fe.homogeneity(hist_diff, diff)
            cluster_sombra_sum = fe.cluster_sombra(hist_sum, mean_sum, sum)
            cluster_prominencia_sum = fe.cluster_prominencia(hist_sum, mean_sum, sum)

            # Agregar las características calculadas a la lista de todas las características
            all_characteristics.extend([
                mean_sum, 
                variance_sum, 
                correlation_sum, 
                contrast_sum, 
                homogeneity_sum, 
                cluster_sombra_sum, 
                cluster_prominencia_sum
            ])

    # Agregar el nombre de la clase a la que pertenecen las características
    all_characteristics.append(file_title[17:20])

    # Generar los nombres de las columnas dinámicamente para los tamaños de ventana de 3 a 25 y para cada columna
    column_names = []
    for col_title in col_titles:
        for window_size in range(3, 26, 2):
            column_names.extend([
                f'{col_title}_mean{window_size}', 
                f'{col_title}_variance{window_size}', 
                f'{col_title}_correlation{window_size}', 
                f'{col_title}_contrast{window_size}', 
                f'{col_title}_homogeneity{window_size}', 
                f'{col_title}_cluster_sombra{window_size}', 
                f'{col_title}_cluster_prominencia{window_size}'
            ])
    
    # Agregar la columna para la clase
    column_names.append('Class')

    # Guardar las características en un archivo CSV
    df = pd.DataFrame([all_characteristics], columns=column_names)

    # Si es el primer archivo, escribe encabezados, si no, solo agrega filas
    df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

    return all_characteristics
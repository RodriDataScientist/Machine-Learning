import matplotlib.pyplot as plt
import numpy as np

# Función para graficar histogramas
def plot_histograms(hist_sum, hist_diff, bin_edges_sum, bin_edges_diff, signal_label):
    plt.figure(figsize=(12, 6))

    # Gráfico para el histograma de la suma
    plt.subplot(1, 2, 1)
    plt.bar(bin_edges_sum[:-1], hist_sum, width=np.diff(bin_edges_sum), edgecolor='black', align='edge')
    plt.title(f'Histograma de Sumas - {signal_label}')
    plt.xlabel('Valor de la suma')
    plt.ylabel('Frecuencia')
    
    # Gráfico para el histograma de la diferencia
    plt.subplot(1, 2, 2)
    plt.bar(bin_edges_diff[:-1], hist_diff, width=np.diff(bin_edges_diff), edgecolor='black', align='edge')
    plt.title(f'Histograma de Diferencias - {signal_label}')
    plt.xlabel('Valor de la diferencia')
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()
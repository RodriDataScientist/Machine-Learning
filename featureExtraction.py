import numpy as np

# Media
def mean(histSum, sum_data):
    return np.sum([sum_data[i] * histSum[min(i, len(histSum) - 1)] for i in range(len(sum_data))])

# Varianza
def variance(histSum, histDif, mean_val, sum_data, diff_data):
    var_sum = np.sum([(sum_data[i] - mean_val) ** 2 * histSum[min(i, len(histSum) - 1)] for i in range(len(sum_data))])
    var_diff = np.sum([(diff_data[i]) ** 2 * histDif[min(i, len(histDif) - 1)] for i in range(len(diff_data))])
    return var_sum + var_diff

# Correlación
def correlation(histSum, histDif, mean_val, sum_data, diff_data):
    corr_sum = np.sum([(sum_data[i] - mean_val) ** 2 * histSum[min(i, len(histSum) - 1)] for i in range(len(sum_data))])
    corr_diff = np.sum([(diff_data[i]) ** 2 * histDif[min(i, len(histDif) - 1)] for i in range(len(diff_data))])
    return corr_sum - corr_diff

# Contraste
def contrast(histDif, diff_data):
    return np.sum([(diff_data[i]) ** 2 * histDif[min(i, len(histDif) - 1)] for i in range(len(diff_data))])

# Homogeneidad
def homogeneity(histDif, diff_data):
    return np.sum([(1 / (1 + diff_data[i])) * histDif[min(i, len(histDif) - 1)] for i in range(len(diff_data))])

# Cluster de Sombra
def cluster_sombra(histSum, mean_val, sum_data):
    return np.sum([(sum_data[i] - mean_val) ** 3 * histSum[min(i, len(histSum) - 1)] for i in range(len(sum_data))])

# Cluster de Prominencia
def cluster_prominencia(histSum, mean_val, sum_data):
    return np.sum([(sum_data[i] - mean_val) ** 4 * histSum[min(i, len(histSum) - 1)] for i in range(len(sum_data))])
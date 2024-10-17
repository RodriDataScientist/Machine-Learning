import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Función para estimar el número adecuado de componentes principales
def estimate_pca_components(input_csv, varianza_objetivo=0.95):
    # Cargar el archivo CSV
    df = pd.read_csv(input_csv)

    # Separar la columna de clase del resto de las características
    features = df.drop(columns=['Class'])
    
    # Estandarizar los datos
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Aplicar PCA sin limitar el número de componentes
    pca = PCA()
    pca.fit(features_scaled)

    # Calcular la varianza explicada acumulada
    varianza_explicada_acumulada = pca.explained_variance_ratio_.cumsum()

    # Graficar la varianza explicada acumulada
    plt.plot(varianza_explicada_acumulada)
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada acumulada vs Número de componentes')
    plt.grid(True)
    plt.show()

    # Determinar el número mínimo de componentes que alcanzan la varianza objetivo
    n_components = next(i for i, total_varianza in enumerate(varianza_explicada_acumulada) if total_varianza >= varianza_objetivo) + 1

    print(f"Número de componentes principales necesarios para conservar al menos {varianza_objetivo*100}% de la varianza: {n_components}")

    return n_components

# Ejemplo de uso
n_components = estimate_pca_components('dataset.csv', varianza_objetivo=0.95)
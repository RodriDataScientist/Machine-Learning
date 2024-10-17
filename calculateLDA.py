import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Función para estimar el número adecuado de componentes de LDA y guardar el CSV con las nuevas características
def estimate_lda_components(input_csv, output_csv="lda_features.csv"):
    # Cargar el archivo CSV
    df = pd.read_csv(input_csv)

    # Separar las características y la clase
    features = df.drop(columns=['Class'])
    labels = df['Class']
    
    # Estandarizar los datos
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Aplicar LDA sin limitar el número de componentes
    lda = LDA()
    features_lda = lda.fit_transform(features_scaled, labels)

    # Obtener la varianza explicada por cada componente
    varianza_explicada = lda.explained_variance_ratio_

    # Calcular la varianza explicada acumulada
    varianza_explicada_acumulada = varianza_explicada.cumsum()

    # Graficar la varianza explicada acumulada
    plt.plot(varianza_explicada_acumulada)
    plt.xlabel('Número de componentes discriminantes')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada acumulada vs Número de componentes (LDA)')
    plt.grid(True)
    plt.show()

    # Determinar el número de componentes disponibles (máximo será el número de clases menos uno)
    n_components = len(varianza_explicada_acumulada)
    print(f"Número de componentes discriminantes disponibles: {n_components}")

    # Guardar las nuevas características generadas por LDA en un archivo CSV
    lda_columns = [f'LD{i+1}' for i in range(features_lda.shape[1])]
    lda_df = pd.DataFrame(features_lda, columns=lda_columns)

    # Agregar la columna de clase de nuevo
    lda_df['Class'] = labels.values

    # Guardar el nuevo dataset con las características LDA
    lda_df.to_csv(output_csv, index=False)
    print(f"Nuevas características guardadas en el archivo: {output_csv}")

    return n_components

# Ejemplo de uso
n_components_lda = estimate_lda_components('dataset.csv', output_csv="dataset_lda.csv")
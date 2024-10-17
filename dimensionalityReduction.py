import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Función para aplicar PCA y reducir dimensionalidad
def apply_pca(input_csv, output_csv, n_components, threshold=0.1):
    df = pd.read_csv(input_csv)

    # Separar la columna de clase del resto de las características
    features = df.drop(columns=['Class'])
    class_column = df['Class']

    # Estandarizar los datos antes de aplicar PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_scaled)

    # Crear un DataFrame con las nuevas características de PCA
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns)

    pca_df['Class'] = class_column.values
    pca_df.to_csv(output_csv, index=False)

    print(f"Archivo con reducción de dimensionalidad guardado en {output_csv}")

    # Mostrar solo las características con mayor contribución (mayor que el umbral)
    feature_contributions = pd.DataFrame(pca.components_, columns=features.columns, index=pca_columns)
    print("Características más relevantes para cada componente principal:\n")

    for component in pca_columns:
        contributions = feature_contributions.loc[component]
        significant_features = contributions[contributions.abs() >= threshold]
        print(f"\n{component}:")
        print(significant_features.sort_values(ascending=False))

apply_pca('dataset.csv', 'dataset_pca.csv', n_components=27, threshold=0.1)
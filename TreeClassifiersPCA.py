import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Lectura y separación del dataset
data = pd.read_csv('dataset_pca.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Obtener la importancia de las características del RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, criterion='entropy')
rf_clf.fit(X, y)
importancias = rf_clf.feature_importances_
indices = np.argsort(importancias)[::-1]

# Seleccionar las 3 características más importantes
top_features = X.columns[indices[:3]]
X_top = X[top_features]

# Graficar en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
clases = np.unique(y)
colormap = plt.colormaps.get_cmap('tab10')

for i, clase in enumerate(clases):
    indices_clase = (y == clase)
    ax.scatter(X_top[indices_clase].iloc[:, 0], 
               X_top[indices_clase].iloc[:, 1], 
               X_top[indices_clase].iloc[:, 2], 
               label=clase, 
               color=colormap(i / len(clases)))

ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel(top_features[2])
ax.set_title('Visualización 3D de las clases usando las características más importantes del Random Forest')
ax.legend()
plt.show()

# Inicializar los clasificadores con parámetros específicos
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, criterion='entropy'),
    "Extra Trees": ExtraTreesClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, criterion='gini')
}

# Crear el esquema de validación cruzada a 5 pliegos
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Entrenar los clasificadores con validación cruzada, medir el tiempo y calcular las métricas para cada pliegue
train_times = {}
results = {}

for name, clf in classifiers.items():
    accuracies = []
    precisions = []
    recalls = []
    cms = None  # Para almacenar la matriz de confusión del último pliegue
    
    start_time = time.time()
    
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Entrenar el clasificador
        clf.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = clf.predict(X_test)
        
        # Calcular métricas
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=1))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=1))
        
        # Guardar la matriz de confusión del último pliegue
        cms = confusion_matrix(y_test, y_pred)
    
    end_time = time.time()
    train_times[name] = end_time - start_time
    
    # Almacenar los resultados (promedios)
    results[name] = {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'confusion_matrix': cms  # Solo del último pliegue
    }

# Mostrar los resultados de precisión, exactitud, tiempos de entrenamiento y matrices de confusión del último pliegue
for name in classifiers.keys():
    print(f"\n{name} - Training Time: {train_times[name]:.2f} seconds")
    print(f"{name} - Mean Accuracy: {results[name]['accuracy']:.2f}")
    print(f"{name} - Mean Precision: {results[name]['precision']:.2f}")
    print(f"{name} - Mean Recall: {results[name]['recall']:.2f}")
    
    # Crear y mostrar la matriz de confusión del último pliegue
    cm = results[name]['confusion_matrix']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'{name} - Confusion Matrix (Last Fold)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
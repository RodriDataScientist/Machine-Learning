import open3d as o3d
import pandas as pd
import numpy as np

def visualizar_nube_puntos(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convertir los datos a un array de numpy
    puntos = df[['X', 'Y', 'Z']].to_numpy()
    
    # Crear un objeto PointCloud de Open3D
    nube_puntos = o3d.geometry.PointCloud()
    nube_puntos.points = o3d.utility.Vector3dVector(puntos)
    
    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([nube_puntos])

visualizar_nube_puntos('Pikachu\Pikachu_1.csv')
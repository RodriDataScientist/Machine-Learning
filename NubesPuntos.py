import numpy as np
import open3d as o3d
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import csv
import random


# Configurar el flujo de datos
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Iniciar la transmisión
pipeline.start(config)

# Streaming loop
frame_count = 0

try:
    while frame_count < 50:
        # Obtener el conjunto de frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convertir a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Crear una nube de puntos
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        xyz = np.asanyarray(vtx).view(np.float32).reshape(-1, 3)
        mask = (xyz[:,0]==0) & (xyz[:,1]==0) & (xyz[:,2] == 0)
        xyz = xyz[~mask]
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1

        # Crear un objeto Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Visualizar la nube de puntos
        o3d.visualization.draw_geometries([pcd])
        axis = 1

        print(f"min: {min(xyz[:, axis])}, max: { max(xyz[:, axis])}")
        xyz_norm = np.linalg.norm(xyz, axis=1)
        max_N2 = np.max(xyz_norm)
        print(xyz.shape)
        dff = random.uniform(.55, .59)
        ind = xyz_norm < dff
        N2vec = xyz[ind, :]
        xyz = N2vec
        pf=random.randrange(30,35)
        valdel=min(xyz[:, axis])*(pf/100)
        ind = xyz[:, axis] > (min(xyz[:, axis])-valdel)
        N2vec = xyz[ind, :]
        print(N2vec.shape)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(N2vec)
        o3d.visualization.draw_geometries([pcd1])

        op = int(input("1: si, 0: no"))
        if (op):
            frame_count += 1
            print(frame_count)
            csv_filename = f'Filament_{frame_count}.csv'
            with open(csv_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['X', 'Y', 'Z'])  # Encabezados
                for point in xyz:
                    csv_writer.writerow(point)
        else:
            continue

finally:
    # Detener la transmisión
    pipeline.stop()
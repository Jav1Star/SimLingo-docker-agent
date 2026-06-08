import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window()
mesh = o3d.geometry.TriangleMesh.create_sphere()
vis.add_geometry(mesh)
print('successfully added geometry')
vis.destroy_window()

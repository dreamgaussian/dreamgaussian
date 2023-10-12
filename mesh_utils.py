import numpy as np
import pymeshlab as pml


def poisson_mesh_reconstruction(points, normals=None):
    """
    Perform mesh reconstruction using the Poisson surface reconstruction algorithm.

    Parameters:
        points (ndarray): A 2D array of shape (N, 3) containing the 3D points.
        normals (ndarray, optional): A 2D array of shape (N, 3) containing the corresponding normals. Defaults to None.

    Returns:
        ndarray: A 2D array of shape (M, 3) containing the vertices of the reconstructed mesh.
        ndarray: A 2D array of shape (P, 3) containing the triangles of the reconstructed mesh.
    """
    # points/normals: [N, 3] np.ndarray

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(
        f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}"
    )

    return vertices, triangles


def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    """
    Decimates the input mesh by reducing the number of vertices.

    Parameters:
        verts (ndarray): An array of vertices.
        faces (ndarray): An array of faces.
        target (int): The target number of vertices.
        backend (str, optional): The backend library to use. Defaults to "pymeshlab".
        remesh (bool, optional): Flag indicating whether to perform remeshing. Defaults to False.
        optimalplacement (bool, optional): Flag indicating whether to use optimal placement. Defaults to True.

    Returns:
        ndarray: The decimated vertices.
        ndarray: The decimated faces.
    """
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.Percentage(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    """
    Clean and repair a mesh.

    This function takes in a set of vertices and faces that define a mesh and applies a series of filters and operations to clean and repair the mesh. The cleaned vertices and faces are then returned.

    Parameters:
        verts (numpy array): The vertices of the mesh.
        faces (numpy array): The faces of the mesh.
        v_pct (float, optional): The percentage of close vertices to merge. Defaults to 1.
        min_f (int, optional): The minimum number of faces for a connected component. Defaults to 64.
        min_d (int, optional): The minimum diameter percentage for a connected component. Defaults to 20.
        repair (bool, optional): Whether to repair the mesh. Defaults to True.
        remesh (bool, optional): Whether to remesh the mesh. Defaults to True.
        remesh_size (float, optional): The target length for remeshing. Defaults to 0.01.

    Returns:
        numpy array: The cleaned vertices of the mesh.
        numpy array: The cleaned faces of the mesh.
    """
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.Percentage(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.AbsoluteValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

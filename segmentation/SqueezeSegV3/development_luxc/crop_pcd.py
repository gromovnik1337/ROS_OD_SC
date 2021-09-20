import open3d as o3d

# Function used to manually crop the point cloud and export the results
def demoCropGeometry(pcd):
    print("Manual cropping visualizer & export")
    print(
        "1) Press 'Y, X or Z' to align geometry with the direction of the respected axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    vis = o3d.visualization.draw_geometries_with_editing([pcd])
    vis.create_window(
        window_name = "Point cloud", 
        width = 720, 
        height = 720, 
        left = 25,
        top = 25 
        )
    # Rendering options
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True

livox_raw_pcd = o3d.io.read_point_cloud("./data/cropped_210920_subsampled.pcd", print_progress = True) # Load the point cloud
demoCropGeometry(livox_raw_pcd)        
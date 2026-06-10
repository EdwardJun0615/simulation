import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


# --- Ship Model Definition ---
# Constants for defining ship geometry
main_deck_height = 15
helipad_height = 10
helipad_end_x = 30
main_body_end_x = 120
bow_start_x = 120
bow_top_end_x = 160
bow_bottom_end_x = 140

# A dictionary defining the vertices of each ship component
ship_model = {
    "main_hull": {
        "v": {
            "sb": np.array([0, -8, 0]),
            "ss": np.array([0, 8, 0]),
            "fb": np.array([main_body_end_x, -8, 0]),
            "fs": np.array([main_body_end_x, 8, 0]),
            "hps": np.array([0, -10, helipad_height]),
            "hss": np.array([0, 10, helipad_height]),
            "hpf": np.array([helipad_end_x, -10, helipad_height]),
            "hsf": np.array([helipad_end_x, 10, helipad_height]),
            "maf": np.array([helipad_end_x, -10, main_deck_height]),
            "msf": np.array([helipad_end_x, 10, main_deck_height]),
            "mff": np.array([main_body_end_x, -10, main_deck_height]),
            "msff": np.array([main_body_end_x, 10, main_deck_height]),
        },
        "critical": False,
    },
    "fwd_engine_room": {
        "v": {
            "blb": np.array([45, -7, 2]), "brb": np.array([45, 7, 2]),
            "flb": np.array([65, -7, 2]), "frb": np.array([65, 7, 2]),
            "blt": np.array([45, -7, 12]), "brt": np.array([45, 7, 12]),
            "flt": np.array([65, -7, 12]), "frt": np.array([65, 7, 12]),
        },
        "critical": True,
    },
    "aft_engine_room": {
        "v": {
            "blb": np.array([75, -7, 2]), "brb": np.array([75, 7, 2]),
            "flb": np.array([95, -7, 2]), "frb": np.array([95, 7, 2]),
            "blt": np.array([75, -7, 12]), "brt": np.array([75, 7, 12]),
            "flt": np.array([95, -7, 12]), "frt": np.array([95, 7, 12]),
        },
        "critical": True,
    },
    "bow": {
        "v": {
            "base_port_bottom": np.array([bow_start_x, -8, 0]),
            "base_stbd_bottom": np.array([bow_start_x, 8, 0]),
            "base_port_top": np.array([bow_start_x, -10, main_deck_height]),
            "base_stbd_top": np.array([bow_start_x, 10, main_deck_height]),
            "tip_bottom": np.array([bow_bottom_end_x, 0, 0]),
            "tip_top": np.array([bow_top_end_x, 0, main_deck_height]),
        },
    },
    "gun": {
        "v": {
            "blb": np.array([125, -2, main_deck_height]), "brb": np.array([125, 2, main_deck_height]),
            "flb": np.array([129, -2, main_deck_height]), "frb": np.array([129, 2, main_deck_height]),
            "blt": np.array([125, -2, main_deck_height + 3]), "brt": np.array([125, 2, main_deck_height + 3]),
            "flt": np.array([129, -2, main_deck_height + 3]), "frt": np.array([129, 2, main_deck_height + 3]),
        },
    },
    "aft_superstructure": {
        "v": {
            "blb": np.array([50, -9, main_deck_height]), "brb": np.array([50, 9, main_deck_height]),
            "flb": np.array([80, -9, main_deck_height]), "frb": np.array([80, 9, main_deck_height]),
            "blt": np.array([50, -7.2, main_deck_height + 6]), "brt": np.array([50, 7.2, main_deck_height + 6]),
            "flt": np.array([80, -7.2, main_deck_height + 6]), "frt": np.array([80, 7.2, main_deck_height + 6]),
        },
    },
    "bridge": {
        "v": {
            "blb": np.array([90, -10, main_deck_height]), "brb": np.array([90, 10, main_deck_height]),
            "flb": np.array([120, -10, main_deck_height]), "frb": np.array([120, 10, main_deck_height]),
            "blt": np.array([90, -6, main_deck_height + 8]), "brt": np.array([90, 6, main_deck_height + 8]),
            "flt": np.array([120, -6, main_deck_height + 8]), "frt": np.array([120, 6, main_deck_height + 8]),
        },
    },
}


# --- CORE ALGORITHMS ---

def get_traversed_units(p1, p2, origin, grid_dims, resolution):
    """An implementation of Amanatides & Woo's fast voxel traversal algorithm."""
    ix, iy, iz = (int(math.floor((c - o) / r)) for c, o, r in zip(p1, origin, resolution))
    end_ix, end_iy, end_iz = (int(math.floor((c - o) / r)) for c, o, r in zip(p2, origin, resolution))

    # Clamp starting and ending indices to be within the grid
    ix = max(0, min(ix, grid_dims[0] - 1))
    iy = max(0, min(iy, grid_dims[1] - 1))
    iz = max(0, min(iz, grid_dims[2] - 1))
    end_ix = max(0, min(end_ix, grid_dims[0] - 1))
    end_iy = max(0, min(end_iy, grid_dims[1] - 1))
    end_iz = max(0, min(end_iz, grid_dims[2] - 1))

    dx, dy, dz = p2 - p1
    stepX = int(np.sign(dx)) if dx != 0 else 0
    stepY = int(np.sign(dy)) if dy != 0 else 0
    stepZ = int(np.sign(dz)) if dz != 0 else 0

    # Distance to next voxel boundary
    next_bound_x = (ix + (1 if stepX > 0 else 0)) * resolution[0] + origin[0]
    next_bound_y = (iy + (1 if stepY > 0 else 0)) * resolution[1] + origin[1]
    next_bound_z = (iz + (1 if stepZ > 0 else 0)) * resolution[2] + origin[2]

    # Time to reach the next voxel boundary
    tMaxX = (next_bound_x - p1[0]) / dx if dx != 0 else float('inf')
    tMaxY = (next_bound_y - p1[1]) / dy if dy != 0 else float('inf')
    tMaxZ = (next_bound_z - p1[2]) / dz if dz != 0 else float('inf')

    # Time to traverse one voxel
    tDeltaX = resolution[0] / abs(dx) if dx != 0 else float('inf')
    tDeltaY = resolution[1] / abs(dy) if dy != 0 else float('inf')
    tDeltaZ = resolution[2] / abs(dz) if dz != 0 else float('inf')

    traversed = [(ix, iy, iz)]
    while (ix, iy, iz) != (end_ix, end_iy, end_iz):
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                ix += stepX
                tMaxX += tDeltaX
            else:
                iz += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                iy += stepY
                tMaxY += tDeltaY
            else:
                iz += stepZ
                tMaxZ += tDeltaZ

        if not (0 <= ix < grid_dims[0] and 0 <= iy < grid_dims[1] and 0 <= iz < grid_dims[2]):
            break
        traversed.append((ix, iy, iz))

    return set(traversed)


def segment_triangle_intersect(p1, p2, tri):
    """Möller–Trumbore intersection algorithm for a line segment and a triangle."""
    v0, v1, v2 = tri
    direction = p2 - p1
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    if -1e-7 < a < 1e-7:
        return False  # Segment is parallel to the triangle plane.

    f = 1.0 / a
    s = p1 - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)

    if v < 0.0 or u + v > 1.0:
        return False

    # Check if the intersection point lies on the line *segment*.
    t = f * np.dot(edge2, q)
    return 0.0 < t < 1.0


# --- SETUP AND ANALYSIS FUNCTIONS ---

def setup_ship_model(model_data, hull_resolution):
    """Voxelizes volumetric parts and prepares triangle geometry for analysis."""
    print("Starting one-time ship model setup...")
    start_time = time.time()

    VOXELIZED_COMPONENTS = ['main_hull', 'fwd_engine_room', 'aft_engine_room']

    all_verts = np.concatenate([np.array(list(comp['v'].values())) for comp in model_data.values()])
    min_coord, max_coord = all_verts.min(axis=0), all_verts.max(axis=0)
    grid_dims = [int(math.ceil((max_coord[i] - min_coord[i]) / hull_resolution[i])) for i in range(3)]

    voxel_map, voxel_cloud, voxel_counts = {}, {}, {}

    # Process critical components first to ensure they aren't overwritten by the hull
    component_order = sorted(model_data.keys(), key=lambda k: model_data.get(k, {}).get('critical', False), reverse=True)

    for name in component_order:
        if name in VOXELIZED_COMPONENTS:
            comp_verts = np.array(list(model_data[name]['v'].values()))
            c_min, c_max = comp_verts.min(axis=0), comp_verts.max(axis=0)

            if not np.all(c_max > c_min):
                continue

            cloud_points = []
            for x in np.arange(c_min[0], c_max[0], hull_resolution[0]):
                for y in np.arange(c_min[1], c_max[1], hull_resolution[1]):
                    for z in np.arange(c_min[2], c_max[2], hull_resolution[2]):
                        ix, iy, iz = int((x - min_coord[0]) / hull_resolution[0]), int((y - min_coord[1]) / hull_resolution[1]), int((z - min_coord[2]) / hull_resolution[2])

                        # If a critical part already claimed this voxel, skip
                        if (ix, iy, iz) in voxel_map:
                            continue

                        voxel_map[(ix, iy, iz)] = name
                        cloud_points.append([x, y, z])

            voxel_cloud[name] = np.array(cloud_points)
            voxel_counts[name] = len(cloud_points)

    voxel_data = {
        "voxel_map": voxel_map, "voxel_cloud": voxel_cloud, "voxel_counts": voxel_counts,
        "grid_dims": grid_dims, "resolution": hull_resolution, "origin": min_coord
    }

    # Prepare triangle geometry for non-voxelized parts
    superstructure_components = {k: v for k, v in model_data.items() if k not in VOXELIZED_COMPONENTS}
    triangles_by_component = {}
    for name, comp_data in superstructure_components.items():
        v, comp_triangles = comp_data['v'], []
        if name == 'bow':
            faces = [
                (v['base_port_bottom'], v['base_stbd_bottom'], v['tip_bottom']), (v['base_port_top'], v['base_stbd_top'], v['tip_top']),
                (v['base_port_bottom'], v['base_port_top'], v['tip_top'], v['tip_bottom']), (v['base_stbd_bottom'], v['base_stbd_top'], v['tip_top'], v['tip_bottom']),
                (v['base_port_bottom'], v['base_stbd_bottom'], v['base_stbd_top'], v['base_port_top'])
            ]
        else: # Generic box shape
            faces = [
                (v['blb'], v['brb'], v['frb'], v['flb']), (v['blt'], v['brt'], v['frt'], v['flt']),
                (v['blb'], v['blt'], v['flt'], v['flb']), (v['brb'], v['brt'], v['frt'], v['frb']),
                (v['blb'], v['brb'], v['brt'], v['blt']), (v['flb'], v['frb'], v['frt'], v['flt'])
            ]

        for face in faces:
            if len(face) == 4:
                comp_triangles.extend([(face[0], face[1], face[2]), (face[0], face[2], face[3])])
            else:
                comp_triangles.append(face)
        triangles_by_component[name] = comp_triangles

    elapsed = time.time() - start_time
    print(f"Ship model setup complete. Took {elapsed:.2f} seconds.")
    return {"voxel_data": voxel_data, "superstructure_triangles_by_comp": triangles_by_component}


def analyze_trajectory(start_point, end_point, precomputed_ship_data):
    """Analyzes a trajectory and generates a detailed hit report."""
    voxel_data = precomputed_ship_data['voxel_data']

    # Find all voxels traversed by the trajectory
    path_indices = get_traversed_units(start_point, end_point, voxel_data['origin'], voxel_data['grid_dims'], voxel_data['resolution'])
    hit_voxel_indices = path_indices.intersection(voxel_data['voxel_map'].keys())

    # Tally hits for each unique voxelized component
    voxel_hit_tally = {}
    for index in hit_voxel_indices:
        component_name = voxel_data['voxel_map'][index]
        voxel_hit_tally[component_name] = voxel_hit_tally.get(component_name, 0) + 1

    # Check for hits on superstructure triangles
    superstructure_hit_report = {}
    hit_triangles = []
    for name, triangles in precomputed_ship_data['superstructure_triangles_by_comp'].items():
        hit_found = False
        for tri in triangles:
            if segment_triangle_intersect(start_point, end_point, tri):
                if not hit_found:
                    hit_found = True
                hit_triangles.append(tri)
        superstructure_hit_report[name] = "HIT" if hit_found else "No Hit"

    # --- Build the final report ---
    report_lines = ["\n--- TRAJECTORY HIT REPORT ---"]

    if voxel_hit_tally:
        report_lines.append("\n--- Voxel Component Damage ---")
        total_counts = voxel_data['voxel_counts']
        for name, hit_count in sorted(voxel_hit_tally.items()):
            total_cells = total_counts.get(name, 0)
            percentage = (hit_count / total_cells * 100) if total_cells > 0 else 0
            critical_marker = " (CRITICAL)" if ship_model.get(name, {}).get('critical') else ""
            report_lines.append(f"{name.replace('_', ' ').title() + ':':<20} {hit_count} of {total_cells} cells destroyed ({percentage:.2f}%){critical_marker}")
    else:
        report_lines.append("\nNo voxel components were hit.")

    report_lines.append("\n--- Superstructure Status ---")
    for name, was_hit in sorted(superstructure_hit_report.items()):
        report_lines.append(f"{name.replace('_', ' ').title() + ':':<20} {was_hit}")
    report_lines.append("---------------------------\n")

    return "\n".join(report_lines), hit_voxel_indices, hit_triangles


# --- VISUALIZATION & MAIN LOOP ---

def visualize_model_and_trajectory(ship_data, start_p=None, end_p=None, hit_voxel_indices=None, hit_triangles=None):
    """Creates a 3D plot of the ship, trajectory, and all hits."""
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Draw voxel clouds for volumetric parts
    voxel_cloud = ship_data['voxel_data']['voxel_cloud']
    if 'main_hull' in voxel_cloud and voxel_cloud['main_hull'].size > 0:
        ax.scatter(voxel_cloud['main_hull'][:, 0], voxel_cloud['main_hull'][:, 1], voxel_cloud['main_hull'][:, 2], c='gray', alpha=0.03, marker='s', s=30, label='Hull (Voxel Cloud)')
    if 'fwd_engine_room' in voxel_cloud and voxel_cloud['fwd_engine_room'].size > 0:
        ax.scatter(voxel_cloud['fwd_engine_room'][:, 0], voxel_cloud['fwd_engine_room'][:, 1], voxel_cloud['fwd_engine_room'][:, 2], c='purple', alpha=0.5, marker='s', s=30, label='Fwd Engine Room')
    if 'aft_engine_room' in voxel_cloud and voxel_cloud['aft_engine_room'].size > 0:
        ax.scatter(voxel_cloud['aft_engine_room'][:, 0], voxel_cloud['aft_engine_room'][:, 1], voxel_cloud['aft_engine_room'][:, 2], c='red', alpha=0.5, marker='s', s=30, label='Aft Engine Room')

    # Draw superstructure as solid faces
    for name, triangles in ship_data['superstructure_triangles_by_comp'].items():
        ax.add_collection3d(art3d.Poly3DCollection(triangles, alpha=0.2, facecolor='blue', edgecolor='gray', lw=0.5))

    # Draw trajectory line
    if start_p is not None and end_p is not None:
        ax.plot([start_p[0], end_p[0]], [start_p[1], end_p[1]], [start_p[2], end_p[2]], 'g-o', lw=2, label='Trajectory')

    # Highlight voxel hits as solid cubes
    if hit_voxel_indices:
        res, origin = ship_data['voxel_data']['resolution'], ship_data['voxel_data']['origin']
        for ix, iy, iz in hit_voxel_indices:
            x, y, z = origin[0] + ix * res[0], origin[1] + iy * res[1], origin[2] + iz * res[2]
            ax.bar3d(x, y, z, res[0], res[1], res[2], color='gold', alpha=0.7, edgecolor='k')

    # Highlight superstructure triangle hits
    if hit_triangles:
        ax.add_collection3d(art3d.Poly3DCollection(hit_triangles, alpha=1.0, facecolor='red', edgecolor='black', lw=1))

    # Set plot aesthetics
    ax.set_xlabel('X (Length)')
    ax.set_ylabel('Y (Width)')
    ax.set_zlabel('Z (Height)')
    ax.set_title('Hybrid Damage Assessment Visualization')
    all_verts_np = np.vstack([v for comp in ship_model.values() for v in comp['v'].values()])
    ax.set_box_aspect((np.ptp(all_verts_np[:, 0]), np.ptp(all_verts_np[:, 1]) * 2, np.ptp(all_verts_np[:, 2]) * 2))
    ax.view_init(elev=25, azim=-75)
    plt.show()


def main():
    """Main loop with lazy initialization of the ship model."""
    ship_data = None
    print("Initializing Damage Assessment Simulation.\nReady for trajectory input.")

    while True:
        try:
            start_input = input("Enter trajectory START point (x,y,z) or type 'exit': ")
            if start_input.lower() == 'exit':
                break
            start_coords = np.array([float(c.strip()) for c in start_input.split(',')])

            end_input = input("Enter trajectory END point (x,y,z): ")
            if end_input.lower() == 'exit':
                break
            end_coords = np.array([float(c.strip()) for c in end_input.split(',')])

            if len(start_coords) != 3 or len(end_coords) != 3:
                print("Invalid format. Please use three comma-separated numbers.")
                continue

            # Lazy initialization: setup the model only on the first run
            if ship_data is None:
                print("\nFirst trajectory. Performing one-time model setup...")
                ship_data = setup_ship_model(ship_model, hull_resolution=[5.0, 2.0, 2.0])

            # Analyze trajectory and print the report
            report, hit_voxels, hit_tris = analyze_trajectory(start_coords, end_coords, ship_data)
            print(report)

            # Display the visualization
            visualize_model_and_trajectory(ship_data, start_coords, end_coords, hit_voxels, hit_tris)

        except (ValueError, IndexError):
            print("\nInvalid input. Please use 'x, y, z' format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    print("Exiting simulation.")


if __name__ == "__main__":
    main()

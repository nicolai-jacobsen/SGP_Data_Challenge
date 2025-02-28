import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

def detect_rounding_for_compound(timeseries_df, compound, compound_course_axis):
    """
    Detects the first rounding of a given compound mark in the timeseries data. The way it detects a rounding, is it
    creates a 'coordinate' system around each compound mark, with the x-axis being across the course axis and the y-axis
    being along the course axis. It then checks if the boat crosses both the x-axis and the y-axis in the correct
    direction to satisfy the conditions for a mark rounding.
    :param timeseries_df: A cut down dataframe containing only data from the start of a race or the last identified mark
    rounding
    :param compound: A dictionary containing information about the compound mark
    :param compound_course_axis: A dictionary containing information about the course axis
    :return:
    """
    comp_name = compound.get("Name", "")
    marks = compound.get("Marks", [])
    bearing = compound_course_axis["Bearing"]
    # Compute the unit vector along the course axis.
    unit_vector = unit_vector_from_bearing(bearing)
    # Compute the perpendicular unit vector
    perp_vector = (-unit_vector[1], unit_vector[0])

    # Convert a boat position into local coordinates relative to a given mark.
    def get_local_coordinates(lat, lon, mark):
        d_lat = float(lat) - mark[0]
        d_lon = float(lon) - mark[1]
        along = d_lat * unit_vector[0] + d_lon * unit_vector[1]
        across = d_lat * perp_vector[0] + d_lon * perp_vector[1]
        return along, across

    # Special case for start and finish compounds. If the boat crosses between the marks, it has rounded.
    if comp_name.startswith("SL") or comp_name.startswith("FL"):
        mark1 = (float(marks[0]["TargetLat"]), float(marks[0]["TargetLng"]))
        mark2 = (float(marks[1]["TargetLat"]), float(marks[1]["TargetLng"]))

        # Project positions along the course axis
        def project_along(lat, lon, ref):
            d_lat = float(lat) - ref[0]
            d_lon = float(lon) - ref[1]
            return d_lat * unit_vector[0] + d_lon * unit_vector[1]

        proj_mark2 = project_along(mark2[0], mark2[1], mark1)
        lower_bound = min(0, proj_mark2)
        upper_bound = max(0, proj_mark2)
        proj_vals = []
        # Iterate over the timeseries data and project each position along the course axis.
        for idx, row in timeseries_df.iterrows():
            proj_val = project_along(row["LATITUDE_GPS_unk"], row["LONGITUDE_GPS_unk"], mark1)
            proj_vals.append(proj_val)
        # Find the first index where the boat's projection crosses the start-finish line.
        rounding_index = None
        for i in range(1, len(proj_vals)):
            # Detect if the boat's projection goes from outside the interval to inside it.
            if not (lower_bound <= proj_vals[i - 1] <= upper_bound) and (lower_bound <= proj_vals[i] <= upper_bound):
                rounding_index = timeseries_df.index[i]
                break
        return rounding_index

    rounding_indices = []
    # Iterate over each mark in the compound and check for a rounding.
    for m in marks:
        mark = (float(m["TargetLat"]), float(m["TargetLng"]))
        local_coords = []
        for idx, row in timeseries_df.iterrows():
            # Convert the boat's position to local coordinates relative to the mark.
            along, across = get_local_coordinates(row["LATITUDE_GPS_unk"],
                                                  row["LONGITUDE_GPS_unk"], mark)
            local_coords.append((along, across))
        rounding_index = None
        # Special case for M1 and WG marks: only inspect positions in the up-course half. The boat cannot round without
        # going above the mark
        if comp_name.startswith('M1') or comp_name.startswith('WG'):
            for i in range(1, len(local_coords)):
                prev_along, prev_across = local_coords[i - 1]
                curr_along, curr_across = local_coords[i]
                # Check if the boat crosses the mark's x and y axis in the correct direction.
                if prev_along > 0 and curr_along > 0:
                    if prev_across * curr_across < 0:
                        rounding_index = timeseries_df.index[i]
                        break
        elif comp_name.startswith('LG'):
            # Only inspect positions in the down-course half
            for i in range(1, len(local_coords)):
                prev_along, prev_across = local_coords[i - 1]
                curr_along, curr_across = local_coords[i]
                # Check if the boat crosses the mark's x and y axis in the correct direction.
                if curr_along < 0 or prev_along < 0:
                    if prev_across * curr_across < 0:
                        rounding_index = timeseries_df.index[i]
                        break
        if rounding_index is not None:
            rounding_indices.append(rounding_index)
    if rounding_indices:
        # If more than one mark yields a rounding, return the earliest event.
        return min(rounding_indices)
    return None

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the compass bearing from point 1 to point 2
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    initial_bearing = math.degrees(math.atan2(x, y))
    return (initial_bearing + 360) % 360


def unit_vector_from_bearing(bearing):
    """
    Given a bearing in degrees, compute a unit vector in (d_lat, d_lon) coordinates.
    """
    rad = math.radians(bearing)
    d_lat = math.cos(rad)
    d_lon = math.sin(rad)
    return (d_lat, d_lon)


def project_point(lat, lon, mark, unit_vector):
    """
    Projects the point (lat, lon) onto the line that passes through the given mark and
    is aligned with the course axis (defined by unit_vector). Instead of measuring
    progress along the course, this function returns the signed perpendicular distance
    from that line. A positive value indicates the boat is on one side of the line,
    while a negative value indicates it is on the other.
    """
    d_lat = float(lat) - mark[0]
    d_lon = float(lon) - mark[1]
    # Compute the perpendicular unit vector
    perp = (-unit_vector[1], unit_vector[0])
    return d_lat * perp[0] + d_lon * perp[1]


def project_point_between_marks(initial, target, current):
    I = np.array(initial)
    T = np.array(target)
    P = np.array(current)

    # Direction vector from initial to target
    d = T - I
    # Direction vector from initial to current
    v = P - I

    # Compute scalar projection factor t
    t = np.dot(v, d) / np.dot(d, d)

    # Compute the projection point
    proj = I + t * d
    return proj


def create_polar_plot(df, angle_col='degrees', speed_col='boat_speed', title="Boat Speeds Polar Plot"):
    """
    Create a polar plot from a DataFrame with angle (in degrees) and speed columns.
    """
    # Convert degrees to radians
    theta = np.deg2rad(df[angle_col])
    r = df[speed_col]

    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r, marker='o', linestyle='-', label='Boat Speed')

    ax.set_title(title)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.legend()

    plt.show()


def ray_boundary_intersection(P0, direction, boundary_points):
    """
    Given a starting point P0 and a ray with unit vector 'direction',
    compute the intersection with the polygon of boundary points.
    boundary_points is a list of coordinate tuples. we convert these to (x,y) as (lon, lat).
    Returns (t, intersection) where t is the distance along the ray, or None if no intersection.
    """
    bpoints = []
    for pt in boundary_points:
        try:
            # Convert to floats
            bpoints.append((float(pt[1]), float(pt[0])))
        except Exception as e:
            print("Skipping boundary point", pt, ":", e)

    intersections = []
    for i in range(len(bpoints) - 1):
        A = np.array(bpoints[i])
        B = np.array(bpoints[i + 1])
        seg = B - A
        rxs = direction[0] * seg[1] - direction[1] * seg[0]
        if np.isclose(rxs, 0):
            continue  # parallel: no intersection
        A_minus_P0 = A - P0
        t = (A_minus_P0[0] * seg[1] - A_minus_P0[1] * seg[0]) / rxs
        u = (A_minus_P0[0] * direction[1] - A_minus_P0[1] * direction[0]) / rxs
        if t >= 0 and 0 <= u <= 1:
            intersections.append((t, P0 + t * direction))
    if intersections:
        t_min, point = min(intersections, key=lambda x: x[0])
        return t_min, point
    else:
        return None


def simulate_path_to_layline(start, target, wind_direction, optimal_twa,
                             boundaries, tolerance=0.005, first_tack=1, safe_margin=0.001):
    """
    Simulate a path from start to target inside given boundaries,
    forcing the boat to maneuver to stay within a safe zone inside the boundary.
    """
    # Convert to cartesian coordinates
    current_pos = np.array([start[1], start[0]])
    target_pos = np.array([target[1], target[0]])

    # Create a Shapely Polygon for the boundary
    # Boundaries is a list of (lat, lon) tuples, so we swap to (lon, lat) for Shapely
    boundary_poly = Polygon([(pt[1], pt[0]) for pt in boundaries])
    # Create a safe zone by buffering inward. We do not want to exceed the boundary
    safe_boundary = boundary_poly.buffer(-safe_margin)

    # Compute tack headings and unit vectors
    tack1_heading = (wind_direction + optimal_twa) % 360
    tack2_heading = (wind_direction - optimal_twa) % 360
    v1 = np.array([np.sin(np.radians(tack1_heading)), np.cos(np.radians(tack1_heading))])
    v2 = np.array([np.sin(np.radians(tack2_heading)), np.cos(np.radians(tack2_heading))])
    directions = {1: v1, 2: v2} # Port/starboard directions
    current_tack = first_tack
    path = [start]
    total_maneuvers = 0

    # While the boat has not reached the target, continue simulating the path.
    while np.linalg.norm(target_pos - current_pos) > tolerance:
        # Get the current direction
        current_direction = directions[current_tack]

        # Ensure the current position is inside the boundary
        # Without this, the boat would end up past the boundary
        current_point = Point(current_pos[0], current_pos[1])
        if not safe_boundary.contains(current_point):
            # Project current_pos to the nearest point inside safe_boundary
            nearest = nearest_points(safe_boundary, current_point)[0]
            # Update current_pos with the nearest point
            current_pos = np.array([nearest.x, nearest.y])

        # Layline check: if a tack directly toward target nearly reaches it, tack on layline
        t_proj = np.dot(target_pos - current_pos, current_direction)
        projection = current_pos + t_proj * current_direction
        error = np.linalg.norm(target_pos - projection)
        if error < tolerance:
            path.append((target_pos[1], target_pos[0]))
            break

        # compute the intersection with the boundary along the current tack
        result = ray_boundary_intersection(current_pos, current_direction, boundaries)
        # If no intersection is found, force a tack
        if result is None:
            # If no intersection is found, force a tack
            current_tack = 2 if current_tack == 1 else 1
            total_maneuvers += 1
            continue
        else:
            t_int, intersection = result

        # Take a step along the current tack – either all the way to the intersection or to the target mark if closer
        step = min(t_int, np.linalg.norm(target_pos - current_pos))
        current_pos = current_pos + step * current_direction
        path.append((current_pos[1], current_pos[0]))

        total_maneuvers += 1
        # Switch tack for the next leg
        current_tack = 2 if current_tack == 1 else 1

    return path, total_maneuvers


def plot_solution(boundaries, path, label, initial, target):
    """
    Plot the solution path on a map with the boundary, start, and target marks.
    :param boundaries: list of boundary points
    :param path: list of path points
    :param label: label for the path
    :param initial: list of initial marks
    :param target: list of target marks
    """
    # Plot boundary
    b_lat = [pt[0] for pt in boundaries] + [boundaries[0][0]]
    b_lon = [pt[1] for pt in boundaries] + [boundaries[0][1]]
    plt.plot(b_lon, b_lat, 'k-', label="Boundary")
    # Plot path
    path_lat = [pt[0] for pt in path]
    path_lon = [pt[1] for pt in path]
    plt.plot(path_lon, path_lat, marker='o', label=label)
    # Plot marks
    for m in initial:
        start_coords = (m['TargetLat'], m['TargetLng'])
        plt.scatter(start_coords[1], start_coords[0], color='green', label="Start")
    for m in target:
        target_coords = (m['TargetLat'], m['TargetLng'])
        plt.scatter(target_coords[1], target_coords[0], color='red', label="Target")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Tacking Solution")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

c1 = np.float64(6400)
c2 = np.float64(1483)
c3 = np.float64(5600)

l0 = np.float64(0.12156646438729327)
h0 = np.float64(0.08843353561270673)
d = l0 + h0

# Maximum sectorial angle (radians)
alpha_max = np.float64(50.62033040986099 * (np.pi / 180))
r_outer = np.float64(0.07)

num_elements = np.int64(64)
pitch = np.float64(0.0006)

num_alpha_points = np.int64(181 * 5)

pipe_offset = 0.001


def find_line_curve_intersection(x_line, y_line, x_curve, y_curve):
    """
    Finds the intersection point between a line (defined by points x_line, y_line)
    and a curve (defined by points x_curve, y_curve).

    Assumes the curve points are ordered generally along the x-axis,
    though strict sorting is not required if the crossing is unique.

    Args:
        x_line (np.ndarray): X-coordinates of points on the line.
        y_line (np.ndarray): Y-coordinates of points on the line.
        x_curve (np.ndarray): X-coordinates of points on the curve.
        y_curve (np.ndarray): Y-coordinates of points on the curve.

    Returns:
        tuple: (x_intersect, y_intersect) coordinates of the intersection point,
               or (None, None) if no intersection is found between segments.
    """
    if len(x_line) < 2 or len(y_line) < 2:
        raise ValueError("Line needs at least two points to be defined.")
    if len(x_curve) < 2 or len(y_curve) < 2:
        raise ValueError("Curve needs at least two points.")
    if len(x_line) != len(y_line) or len(x_curve) != len(y_curve):
        raise ValueError("Coordinate arrays must have the same length.")

    # 1. Find the Line Equation (y = mx + b) using polyfit (linear regression)
    # Check for vertical line first (infinite slope)
    if np.all(np.isclose(x_line, x_line[0])):
        # Vertical line: x = constant
        line_x_const = x_line[0]
        # print(f"Detected vertical line: x = {line_x_const}")

        # Find where the curve crosses this x-value
        # Calculate where x_curve crosses line_x_const
        cross_indices = np.where(np.diff(np.sign(x_curve - line_x_const)) != 0)[0]

        if len(cross_indices) == 0:
             # Check if any curve point lies exactly on the line
             on_line = np.isclose(x_curve, line_x_const)
             if np.any(on_line):
                 idx = np.where(on_line)[0][0]
                #  print(f"Curve point {idx} lies exactly on the vertical line.")
                 return x_curve[idx], y_curve[idx]
             else:
                # print("No intersection found (vertical line does not cross curve segments).")
                return None, None

        # Take the first crossing index
        idx = cross_indices[0]

        # Linear interpolation for y at x = line_x_const, between points idx and idx+1
        x1, x2 = x_curve[idx], x_curve[idx+1]
        y1, y2 = y_curve[idx], y_curve[idx+1]

        # Avoid division by zero if segment is vertical (shouldn't happen if crossing)
        if np.isclose(x1, x2):
            #  print(f"Warning: Curve segment {idx}-{idx+1} is vertical, using midpoint y.")
             y_intersect = (y1 + y2) / 2.0
             return line_x_const, y_intersect

        # Interpolate y
        y_intersect = y1 + (y2 - y1) * (line_x_const - x1) / (x2 - x1)
        # print(f"Intersection found on vertical line between curve points {idx} and {idx+1}")
        return line_x_const, y_intersect

    else:
        # Non-vertical line
        coeffs = np.polyfit(x_line, y_line, 1)
        m, b = coeffs[0], coeffs[1]
        # print(f"Line equation: y = {m:.4f}x + {b:.4f}")

        # 2. Calculate Vertical Differences
        line_y_at_curve_x = m * x_curve + b
        diffs = y_curve - line_y_at_curve_x

        # 3. Find Sign Changes
        sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]

        if len(sign_changes) == 0:
            # Check if any curve point lies exactly on the line
            on_line = np.isclose(diffs, 0)
            if np.any(on_line):
                 idx = np.where(on_line)[0][0]
                #  print(f"Curve point {idx} lies exactly on the line.")
                 return x_curve[idx], y_curve[idx]
            else:
                # print("No intersection found (line does not cross curve segments).")
                # Optional: Could return the point of closest approach
                # min_diff_idx = np.argmin(np.abs(diffs))
                # return x_curve[min_diff_idx], y_curve[min_diff_idx] # Closest point approx
                return None, None

        # Take the first sign change index
        idx = sign_changes[0]
        # print(f"Sign change detected between curve points {idx} and {idx+1}")

        # 4. Interpolate the Intersection
        # We need to find x_intersect such that:
        # y_curve_interpolated(x_intersect) = m * x_intersect + b
        # Let the curve segment be linear between points idx and idx+1
        x1, y1 = x_curve[idx], y_curve[idx]
        x2, y2 = x_curve[idx+1], y_curve[idx+1]

        # Handle vertical curve segment
        if np.isclose(x1, x2):
            # print(f"Curve segment {idx}-{idx+1} is vertical at x={x1}")
            x_intersect = x1
            y_intersect = m * x_intersect + b
            # Check if this y is within the segment's y-bounds
            y_min, y_max = min(y1, y2), max(y1, y2)
            if y_intersect >= y_min - 1e-9 and y_intersect <= y_max + 1e-9: # Use tolerance
                #  print("Intersection found on vertical curve segment.")
                 return x_intersect, y_intersect
            else:
                #  print("Intersection point y is outside the vertical segment bounds.")
                 # This case should ideally not happen if a sign change was correctly detected unless line is also vertical there
                 # Continue searching if multiple sign changes exist? For now, return None.
                 return None, None


        # Equation of the line segment (y = m_seg * x + b_seg)
        m_seg = (y2 - y1) / (x2 - x1)
        b_seg = y1 - m_seg * x1

        # Find x where the two lines intersect: m*x + b = m_seg*x + b_seg
        if np.isclose(m, m_seg):
            # Lines are parallel
            if np.isclose(b, b_seg):
                # print(f"Line and curve segment {idx}-{idx+1} are collinear.")
                # Any point on the overlapping segment is an intersection. Return midpoint?
                # Need to define behaviour here. Returning the segment midpoint for now.
                x_intersect = (x1 + x2) / 2.0
                y_intersect = m * x_intersect + b
                return x_intersect, y_intersect
            else:
                # print(f"Line and curve segment {idx}-{idx+1} are parallel but not collinear.")
                # This shouldn't happen if a sign change was detected across them
                # Maybe try next sign change if available? For now, return None.
                return None, None

        # Solve for x_intersect: (m - m_seg) * x = b_seg - b
        x_intersect = (b_seg - b) / (m - m_seg)

        # Calculate y_intersect using the main line equation
        y_intersect = m * x_intersect + b

        # Optional sanity check: Ensure intersection lies within the segment bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2) # Using y bounds as well

        # Use tolerance for float comparisons
        if (x_intersect >= x_min - 1e-9 and x_intersect <= x_max + 1e-9 and
            y_intersect >= y_min - 1e-9 and y_intersect <= y_max + 1e-9):
            #  print("Intersection confirmed within segment bounds.")
             return x_intersect, y_intersect
        else:
             # This indicates something went wrong, e.g., multiple intersections
             # or issues with the sign change logic for the given data.
            #  print("Warning: Calculated intersection point is outside the segment bounds where sign change was detected.")
            #  print(f"  Segment X: [{x1:.4f}, {x2:.4f}], Intersect X: {x_intersect:.4f}")
            #  print(f"  Segment Y: [{y1:.4f}, {y2:.4f}], Intersect Y: {y_intersect:.4f}")
             # Could try searching other sign changes if len(sign_changes)>1
             return None, None # Or maybe return the calculated point anyway?


def roots_bhaskara(a, b, c):
    sqrt_delta = np.sqrt(np.square(b) - 4 * a * c)
    den = 2 * a
    x1 = (-b + sqrt_delta) / den
    x2 = (-b - sqrt_delta) / den

    return x1, x2


def h_from_alpha(alpha):
    T = l0 / c1 + h0 / c2

    A = (np.square(c1) / np.square(c2)) - 1
    B = 2 * d * np.cos(alpha) - 2 * T * np.square(c1) / c2
    C = (np.square(c1) * np.square(T)) - np.square(d)

    h = roots_bhaskara(A, B, C)[1]

    return h


def dh_from_alpha(alpha):
    T = l0 / c1 + h0 / c2

    A = (np.square(c1) / np.square(c2)) - 1
    # B = 2 * d * np.cos(alpha) - 2 * T * np.square(c1) / c2
    C = (np.square(c1) * np.square(T)) - np.square(d)

    phi_1 = -1 / (2 * A)
    phi_2 = (-2 * T * np.square(c1)) / c2
    phi_3 = 2 * d

    B = phi_2 + phi_3 * np.cos(alpha)

    dB_dAlpha = -phi_3 * np.sin(alpha)
    dB2_dAlpha = 2 * B * dB_dAlpha

    # Equation A.16 in Appendix A.2.1.
    sqrt_aux = np.sqrt(np.square(B) - 4 * A * C)
    dSqrtaux_dAlpha = (1 / (2 * sqrt_aux)) * dB2_dAlpha

    dh_dAlpha = phi_1 * (dB_dAlpha + dSqrtaux_dAlpha)

    return dh_dAlpha


def x_z_from_alpha(alpha):
    h = h_from_alpha(alpha)

    x = h * np.sin(alpha)
    z = h * np.cos(alpha)

    return x, z


def dz_dx_from_alpha(alpha):
    h = h_from_alpha(alpha)
    dh_dAlpha = dh_from_alpha(alpha)

    # Equations (A.19a) and (A.19b) in Appendix A.2.2.
    dz_dAlpha = dh_dAlpha * np.cos(alpha) - h * np.sin(alpha)
    dx_dAlpha = dh_dAlpha * np.sin(alpha) + h * np.cos(alpha)

    return dz_dAlpha, dx_dAlpha


def dzdx_pipe(x_q, r_outer):
    return -x_q / np.sqrt(np.square(r_outer) - np.square(x_q))


def plot_setup(show=True, legend=True):
    transducer_x = np.arange(num_elements) * pitch
    transducer_x = transducer_x - np.mean(transducer_x)
    transducer_y = np.ones_like(transducer_x) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_alpha, z_alpha = x_z_from_alpha(alpha)

    angle_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_points)
    x_pipe = r_outer * np.sin(angle_pipe)
    x_pipe += pipe_offset
    z_pipe = r_outer * np.cos(angle_pipe)

    plt.figure()
    plt.plot(transducer_x, transducer_y, 'o', markersize=1, label="Transducer", color="green")
    plt.plot(x_alpha, z_alpha, label="Refracting surface", color="red")
    plt.plot(x_pipe, z_pipe, label="Pipe", color="blue")
    plt.scatter(0, 0, label="Origin (0, 0)", color="orange")
    plt.scatter(0, d, label="Transducer's center", color="black")
    if legend:
        plt.legend()
    plt.axis("equal")
    if show:
        plt.show()


def refraction(incidence_phi, dzdx, v1, v2):
    """
    dzdx : tuple or ndarray
    """
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = np.arcsin((v2 / v1) * np.sin(theta_1))
    refractive_phi = phi_slope - (np.pi / 2) + theta_2

    return refractive_phi, phi_normal


def reflection(incidence_phi, dzdx):
    if isinstance(dzdx, tuple):
        phi_slope = np.arctan2(dzdx[0], dzdx[1])
    elif isinstance(dzdx, np.ndarray):
        phi_slope = np.arctan(dzdx)
    phi_normal = phi_slope + np.pi / 2
    theta_1 = incidence_phi - (phi_slope + np.pi / 2)
    theta_2 = -theta_1
    reflective_phi = phi_slope - (np.pi / 2) + theta_2
    return reflective_phi, phi_normal


def plot_normal(angle, x, z, scale=0.007, color='purple'):
    normal_dx = np.cos(angle)
    normal_dz = np.sin(angle)

    normal_end_x_pos = x + normal_dx * scale
    normal_end_z_pos = z + normal_dz * scale
    normal_end_x_neg = x - normal_dx * scale
    normal_end_z_neg = z - normal_dz * scale
    plt.plot([normal_end_x_neg, normal_end_x_pos], 
             [normal_end_z_neg, normal_end_z_pos], 
             color, linewidth=1.0, linestyle='-')
    

def plot_line(angle, x, z, scale=0.007, color='purple', x_pos=True, z_pos=True, x_neg=True, z_neg=True):
    normal_dx = np.cos(angle)
    normal_dz = np.sin(angle)

    normal_end_x_pos = x + normal_dx * scale if x_pos else x
    normal_end_z_pos = z + normal_dz * scale if z_pos else z
    normal_end_x_neg = x - normal_dx * scale if x_neg else x
    normal_end_z_neg = z - normal_dz * scale if z_neg else z

    plt.plot([normal_end_x_neg, normal_end_x_pos], 
             [normal_end_z_neg, normal_end_z_pos], 
             color, linewidth=1.0, linestyle='-')


def uhp(x):
    """Projects an angle to the Upper Half Plane [0; pi]"""

    def rhp(x):
        """Projects an angle to the Right Half Plane [-pi/2; pi/2]"""
        x = np.mod(x, np.pi)
        x = x - (x > np.pi / 2) * np.pi
        x = x + (x < -np.pi / 2) * np.pi
        return x

    x = rhp(x)
    x = x + (x < 0) * np.pi
    return x


def shoot_rays(x_a, z_a, x_f, z_f, alpha, plot=True):
    x_p, z_p = x_z_from_alpha(alpha)

    # Equation (B.2) in Appendix B.
    phi_ap = np.arctan2(z_a - z_p, x_a - x_p)

    # Refraction (c1 -> c2)
    d_zh, d_xh = dz_dx_from_alpha(alpha)
    phi_pq, phi_h = refraction(phi_ap, (d_zh, d_xh), c1, c2)

    # Line equation
    a_pq = np.tan(phi_pq)
    b_pq = z_p - a_pq * x_p

    A = np.square(a_pq) + 1
    # Equation (B.11b) in Appendix B. The article is missing the "2".
    # B = 2 * a_pq * b_pq
    # C = np.square(b_pq) - np.square(r_outer)

    B = 2 * (a_pq * b_pq - pipe_offset)
    C = np.square(pipe_offset) + np.square(b_pq) - np.square(r_outer)

    x_q1, x_q2 = roots_bhaskara(A, B, C)
    z_q1 = a_pq * x_q1 + b_pq
    z_q2 = a_pq * x_q2 + b_pq
    mask_upper = z_q1 > z_q2
    x_q = np.where(mask_upper, x_q1, x_q2)
    z_q = np.where(mask_upper, z_q1, z_q2)

    # Reflection in the pipe
    slope_zc_x = dzdx_pipe(x_q, r_outer)

    # If using the function below, there is no need for uhp()
    # phi_l, phi_c = reflection(phi_pq, slope_zc_x)
    phi_l, phi_c = refraction(phi_pq, slope_zc_x, c2, c2)
    phi_l = uhp(phi_l)

    # Line equation
    a_l = np.tan(phi_l)
    b_l = z_q - a_l * x_q

    intersection_x = np.empty_like(alpha)
    intersection_z = np.empty_like(alpha)

    xx = [None] * num_alpha_points
    zz = [None] * num_alpha_points

    for ray in range(num_alpha_points):
        xx[ray] = np.linspace(x_q[ray] - 0.15, x_q[ray] + 0.15, num_alpha_points)
        
        # Utiliza várias coordenadas x (linspace) para encontrar vários valores da reta "de reflexão"
        zz[ray] = a_l[ray] * xx[ray] + b_l[ray]

        x_intersect, y_intersect = find_line_curve_intersection(xx[ray], zz[ray], x_p, z_p)

        intersection_x[ray] = x_intersect
        intersection_z[ray] = y_intersect

    # Refraction (c2 -> c1)
    alpha_intersection = np.arctan2(intersection_x, intersection_z)
    d_z_intersection, d_x_intersection = dz_dx_from_alpha(alpha_intersection)
    phi_last, phi_intersection_incidence = refraction(phi_l, (d_z_intersection, d_x_intersection), c2, c1)

    # Line equation
    a_intersection = np.tan(phi_last)
    b_intersection = intersection_z - a_intersection * intersection_x

    x_in = (z_f - b_intersection) / a_intersection
    z_in = z_f.copy()

    if plot:
        plot_setup(show=False, legend=False)
        plt.title(f"Element at {x_a} m shooting")
        plt.xlim([-0.1, 0.1])
        for idx, ray in enumerate(range(0, num_alpha_points, 10)):
            if idx == 0:
                plt.plot([x_a, x_p[ray]], [z_a, z_p[ray]], "C0", label="Incident ray")
                plt.plot([x_p[ray], x_q[ray]], [z_p[ray], z_q[ray]], "C1", label="Refracted ray (c1->c2)")
                plt.plot([x_q[ray], intersection_x[ray]], [z_q[ray], intersection_z[ray]], "C2", label="Reflected ray")
                plt.plot([intersection_x[ray], x_in[ray]], [intersection_z[ray], z_in[ray]], "C3", label="Refracted ray (c2->c1)")
            else:
                plt.plot([x_a, x_p[ray]], [z_a, z_p[ray]], "C0")
                plt.plot([x_p[ray], x_q[ray]], [z_p[ray], z_q[ray]], "C1")
                plt.plot([x_q[ray], intersection_x[ray]], [z_q[ray], intersection_z[ray]], "C2")
                plt.plot([intersection_x[ray], x_in[ray]], [intersection_z[ray], z_in[ray]], "C3")

            # plt.plot(x_f, z_f, 'o', markersize=0.1)
            # plot_line(phi_last[ray], intersection_x[ray], intersection_z[ray], scale=0.2, x_pos=False, z_pos=False)
            plot_normal(phi_h[ray], x_p[ray], z_p[ray])
            plot_normal(phi_c[ray], x_q[ray], z_q[ray])
            if ray < len(intersection_x):
                plot_normal(phi_intersection_incidence[ray], intersection_x[ray], intersection_z[ray])
        plt.legend()
        plt.show()

    return {
        "lens_1_x": x_p,
        "lens_1_z": z_p,
        "pipe_x": x_q,
        "pipe_z": z_q,
        "lens_2_x": intersection_x,
        "lens_2_z": intersection_z,
        "target_x": x_in,
        "target_z": z_in,
    }


def dist(x1, z1, x2, z2):
    return np.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


if __name__ == "__main__":
    x_a = np.arange(num_elements, dtype=np.float64) * pitch
    x_a = x_a - np.mean(x_a)
    x_aux = list(x_a)
    x_aux.insert(32, np.float64(0.0))
    x_a = np.asarray(x_aux, dtype=np.float64)
    z_a = np.ones_like(x_a) * d

    xf = np.linspace(x_a[0], x_a[-1], num_alpha_points)
    zf = np.ones((num_alpha_points,), dtype=np.float64) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)

    element_idx = 32
    results = {idx: {} for idx in range(num_elements)}
    for m in range(element_idx, element_idx + 1):
        print(f'Element shooting: {m}')
        results[m] = shoot_rays(x_a[m], z_a[m], xf, zf, alpha, plot=True)

    tof_d = {idx: [] for idx in range(num_elements)}
    tof_ray = {idx: [] for idx in range(num_elements)}
    for ray in range(num_alpha_points):
        for elem_idx, elem_x in enumerate(x_a):
            if np.isclose(results[element_idx]["target_x"][ray], elem_x, atol=1e-5):
                tof_d[elem_idx].append(dist(x_a[element_idx], z_a[element_idx], results[element_idx]["lens_1_x"][ray], results[element_idx]["lens_1_z"][ray]) / c1)
                tof_ray[elem_idx].append(alpha[ray])
                tof_d[elem_idx][-1] += dist(results[element_idx]["lens_1_x"][ray], results[element_idx]["lens_1_z"][ray], results[element_idx]["pipe_x"][ray], results[element_idx]["pipe_z"][ray]) / c2
                tof_d[elem_idx][-1] += dist(results[element_idx]["pipe_x"][ray], results[element_idx]["pipe_z"][ray], results[element_idx]["lens_2_x"][ray], results[element_idx]["lens_2_z"][ray]) / c2
                tof_d[elem_idx][-1] += dist(results[element_idx]["lens_2_x"][ray], results[element_idx]["lens_2_z"][ray], results[element_idx]["target_x"][ray], results[element_idx]["target_z"][ray]) / c1
                break

    tof = []
    tof_rray = []
    for elem_idx in range(num_elements):
        if len(tof_d[elem_idx]) > 0:
            tof_ray[elem_idx] = tof_ray[elem_idx][tof_d[elem_idx].index(min(tof_d[elem_idx]))]
            tof_d[elem_idx] = min(tof_d[elem_idx])
        else:
            tof_ray[elem_idx] = -1
            tof_d[elem_idx] = 0
        tof.append(tof_d[elem_idx])
        tof_rray.append(tof_ray[elem_idx])

    tof = np.asarray(tof)
    tof_rray = np.asarray(tof_rray)
    print(tof_rray)

    plt.figure()
    plt.plot(tof, 'o')
    plt.axis('auto')
    plt.xlabel("Element Focused (index)")
    plt.ylabel("Distance (meters)")
    plt.show()

    for m in range(element_idx, element_idx + 1):
        plot_setup(show=False)
        plt.title(f"Element {m} shooting")
        for idx, ray in enumerate(range(0, num_alpha_points, 1)):
            hit_another_elem = False
            for elem_x in x_a:
                if np.isclose(results[m]["target_x"][ray], elem_x, atol=1e-5):
                    hit_another_elem = True
                    break
            if hit_another_elem:
                if idx == 0:
                    plt.plot([x_a[m], results[m]["lens_1_x"][ray]], [z_a[m], results[m]["lens_1_z"][ray]], "C0", label="Incident ray")
                    plt.plot([results[m]["lens_1_x"][ray], results[m]["pipe_x"][ray]], [results[m]["lens_1_z"][ray], results[m]["pipe_z"][ray]], "C1", label="Refracted ray (c1->c2)")
                    plt.plot([results[m]["pipe_x"][ray], results[m]["lens_2_x"][ray]], [results[m]["pipe_z"][ray], results[m]["lens_2_z"][ray]], "C2", label="Reflected ray")
                    plt.plot([results[m]["lens_2_x"][ray], results[m]["target_x"][ray]], [results[m]["lens_2_z"][ray], results[m]["target_z"][ray]], "C3", label="Refracted ray (c2->c1)")
                else:
                    plt.plot([x_a[m], results[m]["lens_1_x"][ray]], [z_a[m], results[m]["lens_1_z"][ray]], "C0")
                    plt.plot([results[m]["lens_1_x"][ray], results[m]["pipe_x"][ray]], [results[m]["lens_1_z"][ray], results[m]["pipe_z"][ray]], "C1")
                    plt.plot([results[m]["pipe_x"][ray], results[m]["lens_2_x"][ray]], [results[m]["pipe_z"][ray], results[m]["lens_2_z"][ray]], "C2")
                    plt.plot([results[m]["lens_2_x"][ray], results[m]["target_x"][ray]], [results[m]["lens_2_z"][ray], results[m]["target_z"][ray]], "C3")

        plt.show()

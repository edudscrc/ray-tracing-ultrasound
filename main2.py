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

roi_angle_max = alpha_max * 0.9
roi_radius_max = r_outer
roi_radius_min = r_outer - 0.04

num_alpha_points = np.int64(181)


def intersect_segments(p1, p2, p3, p4, tol=1e-9):
    """
    Finds the intersection point of two line segments.
    Segment 1: p1 to p2
    Segment 2: p3 to p4
    Returns the intersection point (x, y) if they intersect, otherwise None.
    Uses tolerance for floating point comparisons.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate denominator
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel or collinear
    if abs(den) < tol:
        # Optional: Add check for collinear overlap here if needed
        return None

    # Calculate numerators for parameters t and u
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    # Calculate parameters t and u
    t = t_num / den
    u = u_num / den

    # Check if intersection lies within both segments (using tolerance)
    if (-tol <= t <= 1 + tol) and (-tol <= u <= 1 + tol):
        # Intersection point calculation (using parameter t)
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y)
    else:
        # Intersection point is outside the segments
        return None


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

    # Código como está no Google Colab. Pode ser simplificado pelas equações (A.19a) e (A.19b) do artigo.
    # alpha_ = (np.pi / 2) - alpha
    # dh_dAlpha = -dh_dAlpha
    # dz_dAlpha = dh_dAlpha * np.sin(alpha_) + h * np.cos(alpha_)
    # dx_dAlpha = dh_dAlpha * np.cos(alpha_) - h * np.sin(alpha_)

    # Equations (A.19a) and (A.19b) in Appendix A.2.2.
    dz_dAlpha = dh_dAlpha * np.cos(alpha) - h * np.sin(alpha)
    dx_dAlpha = dh_dAlpha * np.sin(alpha) + h * np.cos(alpha)

    return dz_dAlpha, dx_dAlpha


def dzdx_pipe(x_q, r_outer):
    return -x_q / np.sqrt(np.square(r_outer) - np.square(x_q))


def plot_setup(show=True):
    transducer_x = np.arange(num_elements) * pitch
    transducer_x = transducer_x - np.mean(transducer_x)
    transducer_y = np.ones_like(transducer_x) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_alpha, z_alpha = x_z_from_alpha(alpha)

    angle_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_points)
    x_pipe = r_outer * np.sin(angle_pipe)
    z_pipe = r_outer * np.cos(angle_pipe)

    plt.figure()
    plt.plot(transducer_x, transducer_y, label="Transducer", color="green")
    plt.plot(x_alpha, z_alpha, label="Refracting Surface", color="red")
    plt.plot(x_pipe, z_pipe, label="Pipe", color="blue")
    plt.scatter(0, 0, label="Origin (0, 0)", color="orange")
    plt.scatter(0, d, label="Transducer's Center", color="black")
    plt.legend()
    plt.axis("equal")
    if show:
        plt.show()


# def dist(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def rhp(x):
    '''Projects an angle to the Right Half Plane [-pi/2; pi/2]'''
    x = np.mod(x, np.pi)
    x = x - (x>np.pi/2)*np.pi
    x = x + (x<-np.pi/2)*np.pi
    return x


def uhp(x):
    '''Projects an angle to the Upper Half Plane [0; pi]'''
    x = rhp(x)
    x = x + (x<0)*np.pi
    return x


def shoot_rays(x_a, z_a, x_f, z_f, alpha):
    x_p, z_p = x_z_from_alpha(alpha)

    # Equation (B.2) in Appendix B.
    phi_ap = np.arctan2(z_a - z_p, x_a - x_p)

    # Refraction (c1 -> c2)
    d_zh, d_xh = dz_dx_from_alpha(alpha)
    phi_h = np.arctan2(d_zh, d_xh)
    phi_1 = phi_ap - (phi_h + np.pi / 2)
    phi_2 = np.arcsin((c2 / c1) * np.sin(phi_1))
    # phi_pq = phi_h + (np.pi / 2) - phi_2  # Equation (B.5) in Appendix B.
    # The equation (B.5) above produces incorrect result. The one below produces correct result.
    phi_pq = phi_h - (np.pi / 2) + phi_2

    # If you don't normalize it to the range [0, pi], it will produce incorrect result.
    phi_pq = uhp(phi_pq)

    # Line equation
    a_pq = np.tan(phi_pq)
    b_pq = z_p - a_pq * x_p

    A = np.square(a_pq) + 1
    # Equation (B.11b) in Appendix B. The article is missing the "2".
    B = 2 * a_pq * b_pq
    C = np.square(b_pq) - np.square(r_outer)

    x_q1, x_q2 = roots_bhaskara(A, B, C)
    z_q1 = a_pq * x_q1 + b_pq
    z_q2 = a_pq * x_q2 + b_pq
    mask_upper = z_q1 > z_q2
    x_q = np.where(mask_upper, x_q1, x_q2)
    z_q = np.where(mask_upper, z_q1, z_q2)

    # Refraction (c2 -> c3)
    # slope_zc_x = dzdx_pipe(x_q, r_outer)
    # phi_c = np.arctan(slope_zc_x)
    # phi_3 = phi_pq - (phi_c + np.pi / 2)
    # phi_4 = np.arcsin((c3 / c2) * np.sin(phi_3))
    # # phi_l = phi_c + np.pi / 2 - phi_4  # Equation (B.21) in Appendix B. (Erro de digitação no artigo: phi_4 ao invés de phi_2)
    # # The equation (B.21) above produces incorrect result. The one below produces correct result.
    # phi_l = phi_c - np.pi / 2 + phi_4

    # Reflection in the pipe
    slope_zc_x = dzdx_pipe(x_q, r_outer)
    phi_c = np.arctan(slope_zc_x)
    phi_3 = phi_pq - (phi_c + np.pi / 2)
    phi_4 = -phi_3
    phi_l = (phi_c + np.pi / 2) + phi_4  # Little change in Equation (B.5) for reflection.

    # Line equation
    a_l = np.tan(phi_l)
    b_l = z_q - a_l * x_q

    # Plotting reflection
    x_refl = x_q + 0.05 * np.cos(phi_l)
    z_refl = z_q + 0.05 * np.sin(phi_l)

    # --- Intersection Finding Logic ---
    # Combine coordinates into (N, 2) arrays for easier access
    points_q = np.vstack((x_q, z_q)).T          # Shape (N, 2) - Start points
    points_refl = np.vstack((x_refl, z_refl)).T  # Shape (N, 2) - End points
    points_p = np.vstack((x_p, z_p)).T          # Shape (N, 2) - Curve points
    intersection_x = []
    intersection_z = []
    # Iterate through each line segment (from q to refl)
    for i in range(num_alpha_points):
        p1 = points_q[i]
        p2 = points_refl[i]

        # Iterate through each curve segment (from p[j] to p[j+1])
        for j in range(num_alpha_points - 1):
            p3 = points_p[j]
            p4 = points_p[j + 1]

            # Check for intersection
            intersection_point = intersect_segments(p1, p2, p3, p4)

            if intersection_point is not None:
                intersection_x.append(intersection_point[0])
                intersection_z.append(intersection_point[1])

    intersection_x = np.asarray(intersection_x)
    intersection_z = np.asarray(intersection_z)
    intersection_length = len(intersection_x)

    # Refraction (c2 -> c1)
    phi_l = uhp(phi_l)
    alpha_intersection = np.arctan2(intersection_x, intersection_z)
    d_z_intersection, d_x_intersection = dz_dx_from_alpha(alpha_intersection)
    phi_intersection_incidence = np.arctan2(d_z_intersection, d_x_intersection)
    phi_intersection_1 = phi_l[:intersection_length] - (phi_intersection_incidence + np.pi / 2)
    phi_intersection_2 = np.arcsin((c1 / c2) * np.sin(phi_intersection_1))
    phi_last = phi_intersection_incidence - (np.pi / 2) + phi_intersection_2

    phi_last = uhp(phi_last)

    # Line equation
    a_intersection = np.tan(phi_last)
    b_intersection = intersection_z - a_intersection * intersection_x

    # Plotting last refraction (c2 -> c1)
    # x_last = intersection_x + 0.15 * np.cos(phi_last)
    # z_last = intersection_z + 0.15 * np.sin(phi_last)

    # Closest point to targets (x_f), (z_f)
    a4 = -1 / a_intersection
    b4 = z_f[:intersection_length] - a4 * x_f[:intersection_length]
    x_in = (b4 - b_intersection) / (a_intersection - a4)
    z_in = a_intersection * x_in + b_intersection

    # Helpful to plot normal lines
    normal_line_scale = 0.005

    normal_angle_1 = phi_h + np.pi / 2
    normal_dx_1 = np.cos(normal_angle_1)
    normal_dz_1 = np.sin(normal_angle_1)

    normal_angle_2 = phi_c + np.pi / 2
    normal_dx_2 = np.cos(normal_angle_2)
    normal_dz_2 = np.sin(normal_angle_2)

    normal_angle_3 = phi_intersection_incidence + np.pi / 2
    normal_dx_3 = np.cos(normal_angle_3)
    normal_dz_3 = np.sin(normal_angle_3)

    plot_setup(show=False)
    for ray in range(0, num_alpha_points, 10):
        if ray < len(intersection_x):
            plt.plot([x_a, x_p[ray], x_q[ray], intersection_x[ray], x_in[ray]],
                        [z_a, z_p[ray], z_q[ray], intersection_z[ray], z_in[ray]],
                        "C2")
        else:
            plt.plot([x_a, x_p[ray], x_q[ray], x_refl[ray]],
                        [z_a, z_p[ray], z_q[ray], z_refl[ray]],
                        "C2")

        normal_end_x_pos_1 = x_p[ray] + normal_dx_1[ray] * normal_line_scale
        normal_end_z_pos_1 = z_p[ray] + normal_dz_1[ray] * normal_line_scale
        normal_end_x_neg_1 = x_p[ray] - normal_dx_1[ray] * normal_line_scale
        normal_end_z_neg_1 = z_p[ray] - normal_dz_1[ray] * normal_line_scale
        plt.plot([normal_end_x_neg_1, normal_end_x_pos_1], 
                 [normal_end_z_neg_1, normal_end_z_pos_1], 
                 'r-', linewidth=0.8)
        
        normal_end_x_pos_2 = x_q[ray] + normal_dx_2[ray] * normal_line_scale
        normal_end_z_pos_2 = z_q[ray] + normal_dz_2[ray] * normal_line_scale
        normal_end_x_neg_2 = x_q[ray] - normal_dx_2[ray] * normal_line_scale
        normal_end_z_neg_2 = z_q[ray] - normal_dz_2[ray] * normal_line_scale
        plt.plot([normal_end_x_neg_2, normal_end_x_pos_2], 
                 [normal_end_z_neg_2, normal_end_z_pos_2], 
                 'r-', linewidth=0.8)
        
        if ray < len(intersection_x):
            normal_end_x_pos_3 = intersection_x[ray] + normal_dx_3[ray] * normal_line_scale
            normal_end_z_pos_3 = intersection_z[ray] + normal_dz_3[ray] * normal_line_scale
            normal_end_x_neg_3 = intersection_x[ray] - normal_dx_3[ray] * normal_line_scale
            normal_end_z_neg_3 = intersection_z[ray] - normal_dz_3[ray] * normal_line_scale
            plt.plot([normal_end_x_neg_3, normal_end_x_pos_3], 
                    [normal_end_z_neg_3, normal_end_z_pos_3], 
                    'r-', linewidth=0.8)
    plt.show()

    return {
        "lens_1_x": x_p,
        "lens_1_z": z_p,
        "pipe_x": x_q,
        "pipe_z": z_q,
    }

if __name__ == "__main__":
    x_a = np.arange(num_elements, dtype=np.float64) * pitch
    x_a = x_a - np.mean(x_a)
    z_a = np.ones_like(x_a) * d

    xf = np.linspace(x_a[0], x_a[-1], num_alpha_points)
    zf = np.ones((num_alpha_points,), dtype=np.float64) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)

    results = []
    for m in range(0, num_elements, 10):
        results.append(shoot_rays(x_a[m], z_a[m], xf, zf, alpha))

    # for m in range(num_elements):
    #     plot_setup(show=False)
    #     for ray in range(0, num_alpha_points, 5):
    #         plt.plot([x_a[m], results[m]["lens_1_x"][ray], results[m]["pipe_x"][ray]],
    #                  [z_a[m], results[m]["lens_1_z"][ray], results[m]["pipe_z"][ray]],
    #                  "C2",
    #                  alpha=0.3)
    #     plt.show()

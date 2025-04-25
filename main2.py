import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# Speeds of sound
c1 = 6400
c2 = 1483
c3 = 5600

# Distances 'r' and 'z' for alpha = 0
l0 = 0.12156646438729327
h0 = 0.08843353561270673
d = l0 + h0

# Maximum sectorial angle (radians)
alpha_max = 50.62033040986099 * (np.pi / 180)
r_outer = 0.07

# Number of elements in the transducer
num_elements = 64
# Spacing between the elements
pitch = 0.0006

roi_angle_max = alpha_max * 0.9
roi_radius_max = r_outer
roi_radius_min = r_outer - 0.02

num_roi_angle_points = num_elements


def roots_bhaskara(a, b, c):
    """Computes the roots of the polynomial ax^2 + bx + c = 0"""

    sqrt_delta = np.sqrt(b**2 - 4 * a * c)
    den = 2 * a
    x1 = (-b + sqrt_delta) / den
    x2 = (-b - sqrt_delta) / den

    return x1, x2


def h_l_from_alpha(alpha):
    """Computes the values of z and r of the lens for a given alpha"""

    t = l0 / c1 + h0 / c2
    a = c1**2 / c2**2 - 1
    c = c1**2 * t**2 - d**2

    phi2 = -2 * t * c1**2 / c2
    phi3 = 2 * d

    b = phi2 + phi3 * np.cos(alpha)

    h = roots_bhaskara(a, b, c)[1]
    l = np.sqrt(h**2 + d**2 - 2 * h * d * np.cos(alpha))

    return h, l


def x_z_from_alpha(alpha):
    """Computes the (x,y) coordinates of the lens for a given alpha"""

    h, _ = h_r_from_alpha(alpha)
    x = h * np.sin(alpha)
    z = h * np.cos(alpha)

    return x, z


def d_zh_d_xh_from_alpha(alpha):
    """Computes the slope (dy/dx) of the lens for a given alpha"""

    t = r0 / c_lens + z0 / c_water

    def dzda_from_alpha(alpha):
        """Computes de derivative dz/dalpha for a given alpha"""

        # This function was originally created by Lucas Ocampos
        # at https://colab.research.google.com/drive/18yOqjGbSTO6MvzSMCYePM5cmVU8Ep0aa?usp=sharing

        a = c_lens**2 / c_water**2 - 1
        c = c_lens**2 * t**2 - d**2

        phi1 = -1 / (2 * a)
        phi2 = -2 * t * c_lens**2 / c_water
        phi3 = 2 * d

        b = phi2 + phi3 * np.cos(alpha)

        # db/da
        dbda = -phi3 * np.sin(alpha)
        # db^2/da
        db2da = 2 * b * dbda

        sqrt_delta = np.sqrt(b**2 - 4 * a * c)
        
        # d[sqrt(Δ)]da
        d_sqrtdelta_da = 1 / 2 * 1 / sqrt_delta * db2da
        dzda = phi1 * (dbda + d_sqrtdelta_da)

        return dzda

    _alpha = np.pi / 2 - alpha
    dzda = -dzda_from_alpha(alpha)
    h, _ = h_r_from_alpha(alpha)
    d_zh = dzda * np.sin(_alpha) + h * np.cos(_alpha)
    d_xh = dzda * np.cos(_alpha) - h * np.sin(_alpha)
    
    return d_zh, d_xh


def dxdy_tube(x, r_pipe):
    """Computes the slope of the cilinder (pipe) for a given x"""
    return -x, np.sqrt(r_pipe**2 - x**2)


def reflect(gamma1, dx, dy):
    """Computes the the new angle  after the reflection
    of the first angle with the Law of Reflection"""

    gamma1 = gamma1
    slope = np.arctan2(dx, dy)
    normal = slope + np.pi / 2
    theta1 = gamma1 - normal
    theta2 = -theta1

    # gamma2 = slope - np.pi / 2 - theta2 - theta2  # g1 > 0
    gamma2 = slope - np.pi / 2 + theta2  # g1 < 0

    return gamma2


def distalpha(xc, yc, xf, yf, alpha_lens):
    """For a shot fired from (xc, yc) in the direction x_y_from_alpha(alpha),
    this function computes the two refractions (from c1 to c2 and from
    c2 to c3) and then computes the squared distance between the ray at c3 and
    the "pixel" (xf, yf). A dictionary is returned with the following information:

    (x_lens, y_lens): position where the ray hits the lens
    (x_pipe, y_pipe): position where the ray hits the pipe
    (xin, yin): point on the ray closest to (xf, yf)
    dist: squared distance (xin, yin) to (xf, yf)"""

    # Position where the ray hits the lens
    x_lens, y_lens = x_y_from_alpha(alpha_lens)
    # Angle of the incident ray
    gamma1 = np.arctan2(y_lens - yc, x_lens - xc)
    # Slope of the tangent line at (x_lens, y_lens)
    dy, dx = dydx_from_alpha(alpha_lens)

    # First refraction (c_lens -> c_water)
    gamma2 = snell(gamma1, dy, dx, c_lens, c_water)

    a_line = np.tan(gamma2)
    b_line = y_lens - a_line * x_lens

    a = a_line**2 + 1
    b = 2 * a_line * b_line
    c = b_line**2 - radius_pipe**2
    x_pipe1, x_pipe2 = roots_bhaskara(a, b, c)
    y_pipe1 = a_line * x_pipe1 + b_line
    y_pipe2 = a_line * x_pipe2 + b_line
    x_pipe = np.where(y_pipe1 > y_pipe2, x_pipe1, x_pipe2)
    y_pipe = np.where(y_pipe1 > y_pipe2, y_pipe1, y_pipe2)
    dx_pipe, dy_pipe = dxdy_tube(x_pipe, radius_pipe)
 
    # ==================
    # === REFLECTION ===
    # ==================

    possible_alphas = np.linspace(-alpha_max, alpha_max, num_roi_angle_points)
    possible_x_lens2, possible_y_lens2 = x_y_from_alpha(possible_alphas)

    gamma3 = reflect(gamma2, dx_pipe, dy_pipe)

    a_line2 = np.tan(rhp(gamma3))
    b_line2 = y_pipe - a_line2 * x_pipe

    x_lens2 = np.zeros(x_lens.shape)
    y_lens2 = np.zeros(y_lens.shape)

    x = [None] * num_roi_angle_points
    y = [None] * num_roi_angle_points

    for ray_idx in range(num_roi_angle_points):
        x[ray_idx] = np.linspace(x_pipe[ray_idx] - 0.05, x_pipe[ray_idx] + 0.05, num_roi_angle_points)
        
        # Utiliza várias coordenadas x (linspace) para encontrar vários valores da reta "de reflexão"
        y[ray_idx] = a_line2[ray_idx] * x[ray_idx] + b_line2[ray_idx]

        x_intersect, y_intersect = find_line_curve_intersection(x[ray_idx], y[ray_idx], possible_x_lens2, possible_y_lens2)

        x_lens2[ray_idx] = x_intersect
        y_lens2[ray_idx] = y_intersect

    # ======================
    # === END REFLECTION ===
    # ======================

    alpha_lens2 = np.arctan2(x_lens2, y_lens2)
    dy2, dx2 = dydx_from_alpha(alpha_lens2)

    # Last refraction (c_water -> c_lens)
    gamma4 = snell(gamma3, dy2, dx2, c_water, c_lens)

    a3_line = np.tan(gamma4)
    b3_line = y_lens2 - a3_line * x_lens2

    yin = np.full(xf.shape, yc)
    xin = (yin - b3_line) / a3_line

    # a4 = -1 / a3_line
    # b4 = yf - a4 * xf
    # xin = (b4 - b3_line) / (a3_line - a4)
    # yin = a3_line * xin + b3_line

    dist = (xin - xf) ** 2 + (yin - yf) ** 2

    xx = [None] * num_roi_angle_points
    yy = [None] * num_roi_angle_points

    for ray_idx in range(num_roi_angle_points):
        xx[ray_idx] = np.linspace(x_lens2[ray_idx], x_lens2[ray_idx] + 0.05 if gamma4[ray_idx] <= 0 else - 0.05, num_roi_angle_points)
        
        yy[ray_idx] = a3_line[ray_idx] * xx[ray_idx] + b3_line[ray_idx]

        # print(y_lens2[ray_idx])

        yy_mask = np.where(yy[ray_idx] > yc + 0.005 , np.full(yy[ray_idx].shape, False), np.full(yy[ray_idx].shape, True))
        
        xx[ray_idx] = np.ma.MaskedArray(xx[ray_idx], mask=~yy_mask)
        yy[ray_idx] = np.ma.MaskedArray(yy[ray_idx], mask=~yy_mask)

        # plot_diamond()
        # plt.plot(xx[ray_idx], yy[ray_idx], markersize=1, linewidth=1, color='red')
        # plt.plot([xin[ray_idx]], [yin[ray_idx]], "^")
        # plt.plot([xf[ray_idx]], [yf[ray_idx]], ">")
        # plt.plot(
        #     [xc, x_lens[ray_idx], x_pipe[ray_idx], x_lens2[ray_idx]],
        #     [yc, y_lens[ray_idx], y_pipe[ray_idx], y_lens2[ray_idx]],
        #     "C2",
        #     alpha=0.3,
        # )
        # plt.show()

    plot_diamond()
    # plt.plot(xx, yy, markersize=1, linewidth=1, color='red')
    # plt.plot([xin], [yin], "^")
    # plt.plot([xf], [yf], ">")
    plt.plot([xc], [yc], 'sk')
    for i in np.arange(0, num_roi_angle_points, 10):
        plt.plot(
            [xc, x_lens[i], x_pipe[i], x_lens2[i], xin[i]],
            [yc, y_lens[i], y_pipe[i], y_lens2[i], yin[i]],
            "C2",
            alpha=0.3,
        )
    # plt.plot(
    #     [xc, x_lens[ray_idx], x_pipe[ray_idx], x_lens2[ray_idx]],
    #     [yc, y_lens[ray_idx], y_pipe[ray_idx], y_lens2[ray_idx]],
    #     "C2",
    #     alpha=0.3,
    # )
    plt.show()

    # a4 = -1 / a_line3
    # b4 = yf - a4 * xf
    
    # xin = (b4 - b_line3) / (a_line3 - a4)
    # yin = a4 * xin + b4

    # dist = (xin - xf) ** 2 + (yin - yf) ** 2

    results = {
        "x_lens": x_lens,
        "y_lens": y_lens,
        "x_pipe": x_pipe,
        "y_pipe": y_pipe,
        "x_lens2": x_lens2,
        "y_lens2": y_lens2,
        "xin": xin,
        "yin": yin,
        "dist": dist,
    }

    return results


# def distance_and_derivatives(xc, yc, xf, yf, alpha_lens, eps=1e-5):
#     """Computes the squared distance using distalpha as well as the first and
#     second derivatives of the squared distance with relation to alpha."""

#     results = distalpha(xc, yc, xf, yf, alpha_lens)

#     distance_minus = distalpha(xc, yc, xf, yf, alpha_lens - eps)["dist"]
#     distance = results["dist"]
#     distance_plus = distalpha(xc, yc, xf, yf, alpha_lens + eps)["dist"]

#     # Finite Differences (Central Difference)
#     first_derivative = (distance_plus - distance_minus) * 0.5 / eps
#     second_derivative = (distance_minus - 2 * distance + distance_plus) / eps**2

#     return results, first_derivative, second_derivative


# def newton(xc, yc, xf, yf, alpha_initial_guess=None, iter=6):
#     """Uses the Newton-Raphson method.

#     Computes the direction in which the transducer at (xc, yc) should fire in order to hit the "pixel" at (xf, yf).

#     Parameters
#     ----------
#     xc : float
#     	x-coordinate of the transducer's element.
#     yc : float
#     	y-coordinate of the transducer's element.
#     xf : ndarray
#     	Array with x-coordinate of each target.
#     yf : ndarray
#     	Array with y-coordinate of each target.
#     alpha_initial_guess : ndarray
#     	Array with alpha values.
#     iter : int
#     	Number of iterations for Newton-Raphson method.

#     Returns
#     -------
#     dic : dict
# 		(x_lens, y_lens): position where the ray hits the lens.
# 		(x_pipe, y_pipe): position where the ray hits the pipe.
# 		(xin, yin): point on the ray closest to (xf, yf).
# 		dist: squared distance (xin, yin) to (xf, yf) (should be close to zero).
# 		maxdist: maximum squared distance (assuming an array of pixels was passed).
# 		maxdist: maximum squared distance (assuming an array of pixels was passed).
#     """

#     if alpha_initial_guess is None:
#         alpha_initial_guess = np.arctan2(xf, yf)

#     maxdist = []
#     mindist = []
#     for _ in range(iter):
#         results, first_derivative, _ = distance_and_derivatives(xc, yc, xf, yf, alpha_initial_guess, eps=1e-4)

#         alpha_initial_guess -= results["dist"] / first_derivative
#         alpha_initial_guess[alpha_initial_guess > alpha_max] = roi_angle_max
#         alpha_initial_guess[alpha_initial_guess < -alpha_max] = -roi_angle_max

#         maxdist.append(results["dist"].max())
#         mindist.append(results["dist"].min())

#     results["maxdist"] = maxdist
#     results["mindist"] = mindist

#     return results


# def newton_batch(xc, yc, xf, yf, iter=6):
#     """Calls the function newton() one time for each transducer element.

#         The set of angles found for a given element are used as initial guess for the next one.
#         Starts from the center of the transducer.

#         Parameters
#         ----------
#         xc : ndarray
# 			Array with x-coordinate of each transducer's element.
#         yc : ndarray
# 			Array with y-coordinate of each transducer's element.
#         xf : ndarray
# 			Array with x-coordinate of each target.
#         yf : ndarray
# 			Array with y-coordinate of each target.
#         iter : int
# 			Number of iterations for Newton-Raphson method.

#         Returns
#         -------
#         results : dict
# 			(x_lens, y_lens): position where the ray hits the lens.
# 			(x_pipe, y_pipe): position where the ray hits the pipe.
# 			(xin, yin): point on the ray closest to (xf, yf).
# 			dist: squared distance (xin, yin) to (xf, yf) (should be close to zero).
# 			maxdist: maximum squared distance (assuming an array of pixels was passed).
#         """

#     transducer_center = num_elements // 2

#     results = [None] * num_elements

#     results[transducer_center] = newton(
#         xc=xc[transducer_center],
#         yc=yc[transducer_center],
#         xf=xf,
#         yf=yf,
#         iter=iter
#     )
#     results[transducer_center - 1] = newton(
#         xc=xc[transducer_center - 1],
#         yc=yc[transducer_center - 1],
#         xf=xf,
#         yf=yf,
#         iter=iter
#     )

#     for i in range(1, transducer_center):
#         i_minus = transducer_center - i - 1
#         i_plus = transducer_center + i

# 		# Newton-Raphson method for i_plus
#         alpha_initial_guess_plus = np.arctan2(results[i_plus - 1]["x_lens"], results[i_plus - 1]["y_lens"])
#         results[i_plus] = newton(
#             xc=xc[i_plus],
#             yc=yc[i_plus],
#             xf=xf,
#             yf=yf,
#             alpha_initial_guess=alpha_initial_guess_plus,
#             iter=iter
#         )
#         bad_indices = results[i_plus]["dist"] > 1e-10
#         num_bad_indices = np.count_nonzero(bad_indices)
#         if num_bad_indices > 0:
#             print(f"Bad indices found at {i_plus}: {num_bad_indices}")

# 		# Newton-Raphson method for i_minus
#         alpha_initial_guess_minus = np.arctan2(results[i_minus + 1]["x_lens"], results[i_minus + 1]["y_lens"])
#         results[i_minus] = newton(
#             xc=xc[i_minus],
#             yc=yc[i_minus],
#             xf=xf,
#             yf=yf,
#             alpha_initial_guess=alpha_initial_guess_minus,
#             iter=iter
#         )
#         bad_indices = results[i_minus]["dist"] > 1e-10
#         num_bad_indices = np.count_nonzero(bad_indices)
#         if num_bad_indices > 0:
#             print(f"Bad indices found at {i_minus}: {num_bad_indices}")

#         print(f"Computed {i * 2} elements (of {num_elements})")

#     return results


def plot_diamond():
    """Plots the main features of the setup."""

    num_alpha_points = num_roi_angle_points
    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_lens, y_lens = x_y_from_alpha(alpha)

    num_alpha_pipe_points = num_roi_angle_points
    alpha_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_pipe_points)
    x_pipe = radius_pipe * np.sin(alpha_pipe)
    y_pipe = radius_pipe * np.cos(alpha_pipe)

    plt.figure()
    plt.plot(x_lens, y_lens, "-k")
    plt.plot(x_pipe, y_pipe, "-C0")
    plt.plot([0], [0], "or")
    # plt.plot([0], [d], "sk")
    plt.axis("equal")


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def snell(incidence_inclination, dz, dx, c1, c2):
    """
    slope_inclination : inclination of the 'divisor' between the two mediums.
    theta_1 : angle of incidence.
    theta_2 : refraction angle.
    refraction_phi : inclination of the refracted segment.
    """

    slope_inclination = np.arctan2(dz, dx)
    theta_1 = incidence_inclination - slope_inclination - (np.pi / 2)
    theta_2 = np.arcsin((c2 / c1) * np.sin(theta_1))
    refraction_phi = slope_inclination + (np.pi / 2) - theta_2
    return refraction_phi


def shoot_rays(xc, yc, xf, yf):
    """
    (x_p, z_p) : cartesian coordinates of point P in lens.
    (d_zh, d_xh) : slope of the curve h(alpha).
    phi_ap : inclination of segment ap.
    phi_pq : inclination of segment pq.
    (a_pq, b_pq) : coefficients of the line equation for segment pq.
    (x_q, z_q) : cartesian coordinates of point Q in coupling medium.
    z_pq : line equation for segment pq.
    """
    alpha = np.linspace(-alpha_max, alpha_max, num_roi_angle_points)
    
    x_p, z_p = x_z_from_alpha(alpha)
    d_zh, d_xh = d_zh_d_xh_from_alpha(alpha)

    phi_ap = np.arctan2(yc - z_p, xc - x_p)

    phi_pq = snell(phi_ap, d_zh, d_xh, c1, c2)

    a_pq = np.tan(phi_pq)
    b_pq = z_p - a_pq * x_p

    A = 1 + a_pq ** 2
    B = 2 * a_pq * b_pq
    C = b_pq ** 2 - r_outer ** 2

    x_q1, x_q2 = roots_bhaskara(A, B, C)
    z_q1 = a_pq * x_q1 + b_pq
    z_q2 = a_pq * x_q2 + b_pq

    mask_upper = (z_q1 > z_q2)

    x_q = np.where(mask_upper, x_q1, x_q2)

    z_pq = a_pq * x_q + b_pq

    


if __name__ == "__main__":
    # plot_diamond()
    # plt.show()

    # 'xc' and 'yc' are arrays of the positions (x, y) of each transducer's element
    xc = np.arange(num_elements) * pitch
    xc = xc - np.mean(xc)
    yc = np.ones_like(xc) * d

    # af = np.linspace(-roi_angle_max, roi_angle_max, num_roi_angle_points)
    # rf = np.linspace(roi_radius_min, roi_radius_max, 1)
    # Af, Rf = np.meshgrid(af, rf)
    # Af = Af.flatten()
    # Rf = Rf.flatten()

    # # 'xf' and 'yf' are arrays of the positions (x, y) of each target (the suffix 'f' means fire)
    # xf = Rf * np.sin(Af)
    # yf = Rf * np.cos(Af)

    xf = copy.deepcopy(xc)
    yf = copy.deepcopy(yc)

    start = time.time()
    results = newton_batch(xc, yc, xf, yf, iter=20)
    end = time.time()
    print(f"Elapsed time - newton_batch: {end - start} seconds.")

    # idx_element = 32

    # plt.figure()
    # plt.semilogy(results[idx_element]["maxdist"], "o-")
    # plt.semilogy(results[idx_element]["mindist"], "o-")
    # plt.grid()
    # plt.xlabel("iteration")
    # plt.ylabel("Distances")
    # plt.legend(["Max distance", "Min distance"])
    # plt.title(f"Newton algorithm convergence fof element {idx_element}")
    # plt.show()

    # tof = dist(
    #     xc[idx_element],
    #     yc[idx_element],
    #     results[idx_element]["x_lens"],
    #     results[idx_element]["y_lens"],
    # ) / c_lens

    # tof += dist(
    #     results[idx_element]["x_lens"],
    #     results[idx_element]["y_lens"],
    #     results[idx_element]["x_pipe"],
    #     results[idx_element]["y_pipe"],
    # ) / c_water

    # tof += dist(
    #     results[idx_element]["x_pipe"],
    #     results[idx_element]["y_pipe"],
    #     xf,
    #     yf,
    # ) / c_pipe

    # tof = tof.reshape((len(rf), len(af)))
    
    # plt.figure()
    # plt.imshow(tof)
    # plt.colorbar()
    # plt.axis("auto")
    # plt.title(f"Times of flight for element {idx_element}")
    # plt.show()

    for idx_element in range(0, num_elements):
        plot_diamond()
        plt.plot(xc, yc, ".k")
        for i in np.arange(0, num_roi_angle_points, 5):
            plt.plot(
                [xc[idx_element], results[idx_element]["x_lens"][i], results[idx_element]["x_pipe"][i], results[idx_element]["x_lens2"][i], results[idx_element]["xin"][i]],
                [yc[idx_element], results[idx_element]["y_lens"][i], results[idx_element]["y_pipe"][i], results[idx_element]["y_lens2"][i], results[idx_element]["yin"][i]],
                "C2",
                alpha=0.3,
            )
        plt.show()

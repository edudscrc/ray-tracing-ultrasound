import numpy as np
import matplotlib.pyplot as plt
import time
import copy

c1 = np.int64(6400)
c2 = np.int64(1483)
c3 = np.int64(5600)

l0 = np.float64(0.12156646438729327)
h0 = np.float64(0.08843353561270673)
d = l0 + h0

# Maximum sectorial angle (radians)
alpha_max = np.float64(50.62033040986099 * (np.pi / 180))
r_outer = np.float64(0.07)

num_elements = np.int64(64)
pitch = np.float64(0.0006)

# roi_angle_max = np.float64(alpha_max * 0.9)

num_alpha_points = 64


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

    dz_dAlpha = dh_dAlpha * np.cos(alpha) - h * np.sin(alpha)
    dx_dAlpha = dh_dAlpha * np.sin(alpha) + h * np.cos(alpha)

    return dz_dAlpha, dx_dAlpha


def dzdx_pipe(x_q, r_outer):
    return -x_q / np.sqrt(np.square(r_outer) - np.square(x_q))


def plot_setup():
    transducer_x = np.arange(num_elements) * pitch
    transducer_x = transducer_x - np.mean(transducer_x)
    transducer_y = np.ones_like(transducer_x) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_alpha, z_alpha = x_z_from_alpha(alpha)

    angle_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_points)
    x_pipe = r_outer * np.sin(angle_pipe)
    z_pipe = r_outer * np.cos(angle_pipe)

    plt.figure()
    plt.plot(transducer_x, transducer_y, label="Transducer")
    plt.plot(x_alpha, z_alpha, label="Refracting Surface")
    plt.plot(x_pipe, z_pipe, label="Pipe")
    plt.scatter(0, 0, label="Origin (0, 0)")
    plt.scatter(0, d, label="Transducer's Center")
    plt.legend()
    plt.show()


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


# def shoot_rays(xc, yc, xf, yf):
#     """
#     (x_p, z_p) : cartesian coordinates of point P in lens.
#     (d_zh, d_xh) : slope of the curve h(alpha).
#     phi_ap : inclination of segment ap.
#     phi_pq : inclination of segment pq.
#     (a_pq, b_pq) : coefficients of the line equation for segment pq.
#     (x_q, z_q) : cartesian coordinates of point Q in coupling medium.
#     z_pq : line equation for segment pq.
#     """
#     alpha = np.linspace(-alpha_max, alpha_max, num_roi_angle_points)
    
#     x_p, z_p = x_z_from_alpha(alpha)
#     d_zh, d_xh = d_zh_d_xh_from_alpha(alpha)

#     phi_ap = np.arctan2(yc - z_p, xc - x_p)

#     phi_pq = snell(phi_ap, d_zh, d_xh, c1, c2)

#     a_pq = np.tan(phi_pq)
#     b_pq = z_p - a_pq * x_p

#     A = 1 + a_pq ** 2
#     B = 2 * a_pq * b_pq
#     C = b_pq ** 2 - r_outer ** 2

#     x_q1, x_q2 = roots_bhaskara(A, B, C)
#     z_q1 = a_pq * x_q1 + b_pq
#     z_q2 = a_pq * x_q2 + b_pq

#     mask_upper = (z_q1 > z_q2)

#     x_q = np.where(mask_upper, x_q1, x_q2)

#     z_pq = a_pq * x_q + b_pq

    


if __name__ == "__main__":
    plot_setup()
    # plot_diamond()
    # plt.show()

    # 'xc' and 'yc' are arrays of the positions (x, y) of each transducer's element
    # xc = np.arange(num_elements) * pitch
    # xc = xc - np.mean(xc)
    # yc = np.ones_like(xc) * d

    # af = np.linspace(-roi_angle_max, roi_angle_max, num_roi_angle_points)
    # rf = np.linspace(roi_radius_min, roi_radius_max, 1)
    # Af, Rf = np.meshgrid(af, rf)
    # Af = Af.flatten()
    # Rf = Rf.flatten()

    # # 'xf' and 'yf' are arrays of the positions (x, y) of each target (the suffix 'f' means fire)
    # xf = Rf * np.sin(Af)
    # yf = Rf * np.cos(Af)

    # xf = copy.deepcopy(xc)
    # yf = copy.deepcopy(yc)

    # start = time.time()
    # results = newton_batch(xc, yc, xf, yf, iter=20)
    # end = time.time()
    # print(f"Elapsed time - newton_batch: {end - start} seconds.")

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

    # for idx_element in range(0, num_elements):
    #     plot_diamond()
    #     plt.plot(xc, yc, ".k")
    #     for i in np.arange(0, num_roi_angle_points, 5):
    #         plt.plot(
    #             [xc[idx_element], results[idx_element]["x_lens"][i], results[idx_element]["x_pipe"][i], results[idx_element]["x_lens2"][i], results[idx_element]["xin"][i]],
    #             [yc[idx_element], results[idx_element]["y_lens"][i], results[idx_element]["y_pipe"][i], results[idx_element]["y_lens2"][i], results[idx_element]["yin"][i]],
    #             "C2",
    #             alpha=0.3,
    #         )
    #     plt.show()

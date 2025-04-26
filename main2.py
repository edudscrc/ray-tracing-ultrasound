import numpy as np
import matplotlib.pyplot as plt
import time
import copy

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

    # Google Colab (pode ser simplificado)
    # alpha_ = (np.pi / 2) - alpha
    # dh_dAlpha = -dh_dAlpha
    # dz_dAlpha = dh_dAlpha * np.sin(alpha_) + h * np.cos(alpha_)
    # dx_dAlpha = dh_dAlpha * np.cos(alpha_) - h * np.sin(alpha_)

    # Igual ao artigo
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


# def snell(incidence_inclination, dz, dx, c1, c2):
#     """
#     slope_inclination : inclination of the 'divisor' between the two mediums.
#     theta_1 : angle of incidence.
#     theta_2 : refraction angle.
#     refraction_phi : inclination of the refracted segment.
#     """

#     slope_inclination = np.arctan2(dz, dx)
#     theta_1 = incidence_inclination - slope_inclination - (np.pi / 2)
#     theta_2 = np.arcsin((c2 / c1) * np.sin(theta_1))
#     refraction_phi = slope_inclination + (np.pi / 2) - theta_2
#     return refraction_phi


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

    phi_ap = np.arctan2(z_a - z_p, x_a - x_p)  # igual ao artigo
    # Código no colab (pode ser simplificado):
    # phi_ap = np.arctan2(z_p - z_a, x_p - x_a)
    # phi_ap = phi_ap + (phi_ap < 0) * np.pi

    # First refraction (Snell's Law)
    d_zh, d_xh = dz_dx_from_alpha(alpha)
    phi_h = np.arctan2(d_zh, d_xh)

    normal_angle_first = phi_h + np.pi / 2
    normal_dx_first = np.cos(normal_angle_first)
    normal_dz_first = np.sin(normal_angle_first)

    phi_1 = phi_ap - (phi_h + np.pi / 2)
    phi_2 = np.arcsin((c2 / c1) * np.sin(phi_1))
    # phi_pq = phi_h + (np.pi / 2) - phi_2  # igual ao artigo
    phi_pq = phi_h - (np.pi / 2) + phi_2

    # phi_pq = uhp(phi_pq)  # isso muda tudo
    a_pq = np.tan(phi_pq)
    b_pq = z_p - a_pq * x_p

    A = np.square(a_pq) + 1
    B = 2 * a_pq * b_pq
    C = np.square(b_pq) - np.square(r_outer)

    x_q1, x_q2 = roots_bhaskara(A, B, C)
    z_q1 = a_pq * x_q1 + b_pq
    z_q2 = a_pq * x_q2 + b_pq

    mask_upper = z_q1 > z_q2
    x_q = np.where(mask_upper, x_q1, x_q2)
    z_q = np.where(mask_upper, z_q1, z_q2)

    # Second refraction (Snell's Law)
    slope_zc_x = dzdx_pipe(x_q, r_outer)
    phi_c = np.arctan(slope_zc_x)
    # phi_c = rhp(phi_c)

    normal_angle_second = phi_c + np.pi / 2
    normal_dx_second = np.cos(normal_angle_second)
    normal_dz_second = np.sin(normal_angle_second)

    phi_3 = phi_pq - (phi_c + np.pi / 2)
    phi_4 = np.arcsin((c3 / c2) * np.sin(phi_3))
    # phi_l = phi_c + np.pi / 2 - phi_4  # Equation B.21 in Appendix B. (Errado no artigo: phi_4 ao invés de phi_2)
    phi_l = phi_c - np.pi / 2 + phi_4

    a_l = np.tan(phi_l)
    b_l = z_q - a_l * x_q

    a4 = -1 / a_l
    b4 = z_f - a4 * x_f
    x_in = (b4 - b_l) /(a_l - a4)
    z_in = a_l * x_in + b_l

    scale = 0.01

    plot_setup(show=False)
    for ray in range(0, len(alpha), 5):
        plt.plot([x_a, x_p[ray], x_q[ray], x_in[ray]],
                    [z_a, z_p[ray], z_q[ray], z_in[ray]],
                    "C2")

        normal_end_x_pos_first = x_p[ray] + normal_dx_first[ray] * scale
        normal_end_z_pos_first = z_p[ray] + normal_dz_first[ray] * scale
        normal_end_x_neg_first = x_p[ray] - normal_dx_first[ray] * scale
        normal_end_z_neg_first = z_p[ray] - normal_dz_first[ray] * scale
        plt.plot([normal_end_x_neg_first, normal_end_x_pos_first], 
                 [normal_end_z_neg_first, normal_end_z_pos_first], 
                 'r-', linewidth=1)
        
        normal_end_x_pos_second = x_q[ray] + normal_dx_second[ray] * scale
        normal_end_z_pos_second = z_q[ray] + normal_dz_second[ray] * scale
        normal_end_x_neg_second = x_q[ray] - normal_dx_second[ray] * scale
        normal_end_z_neg_second = z_q[ray] - normal_dz_second[ray] * scale
        plt.plot([normal_end_x_neg_second, normal_end_x_pos_second], 
                 [normal_end_z_neg_second, normal_end_z_pos_second], 
                 'r-', linewidth=1)
    plt.show()

    return {
        "lens_1_x": x_p,
        "lens_1_z": z_p,
        "pipe_x": x_q,
        "pipe_z": z_q,
        # ""
    }

if __name__ == "__main__":
    # plot_setup()

    x_a = np.arange(num_elements, dtype=np.float64) * pitch
    x_a = x_a - np.mean(x_a)
    z_a = np.ones_like(x_a) * d

    af = np.linspace(-roi_angle_max, roi_angle_max, num_alpha_points)
    rf = np.linspace(roi_radius_min, roi_radius_max, 1)
    Af, Rf = np.meshgrid(af, rf)
    Af = Af.flatten()
    Rf = Rf.flatten()

    # 'xf' and 'yf' are arrays of the positions (x, y) of each target (the suffix 'f' means fire)
    xf = Rf * np.sin(Af)
    zf = Rf * np.cos(Af)

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)

    results = []
    for m in range(num_elements):
        results.append(shoot_rays(x_a[m], z_a[m], xf, zf, alpha))

    for m in range(num_elements):
        plot_setup(show=False)
        for ray in range(0, num_alpha_points, 5):
            plt.plot([x_a[m], results[m]["lens_1_x"][ray], results[m]["pipe_x"][ray]],
                     [z_a[m], results[m]["lens_1_z"][ray], results[m]["pipe_z"][ray]],
                     "C2",
                     alpha=0.3)
        plt.show()

    # z_f, x_f = copy.deepcopy(z_a), copy.deepcopy(x_a)



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

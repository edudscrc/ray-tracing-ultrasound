import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

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


def x_z_from_alpha(alpha):
    h = h_from_alpha(alpha)

    x = h * np.sin(alpha)
    z = h * np.cos(alpha)

    return x, z


def plot_setup(show=True, legend=True):
    transducer_x = np.arange(num_elements) * pitch
    transducer_x = transducer_x - np.mean(transducer_x)
    transducer_y = np.ones_like(transducer_x) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_alpha, z_alpha = x_z_from_alpha(alpha)

    angle_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_points)
    x_pipe = r_outer * np.sin(angle_pipe)
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


def dist(x1, z1, x2, z2):
    return np.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def fermat(params, x_a, z_a, x_f, z_f, c1, c2):
    alpha_1, x_p, z_p, alpha_2 = params

    x_1, z_1 = x_z_from_alpha(alpha_1)
    x_2, z_2 = x_z_from_alpha(alpha_2)

    dist_1 = np.sqrt(np.square(x_a - x_1) + np.square(z_a - z_1))
    dist_2 = np.sqrt(np.square(x_1 - x_p) + np.square(z_1 - z_p))
    dist_3 = np.sqrt(np.square(x_p - x_2) + np.square(z_p - z_2))
    dist_4 = np.sqrt(np.square(x_2 - x_f) + np.square(z_2 - z_f))

    tof_1 = dist_1 / c1
    tof_2 = dist_2 / c2
    tof_3 = dist_3 / c2
    tof_4 = dist_4 / c1

    total_tof = tof_1 + tof_2 + tof_3 + tof_4

    print(f"{total_tof = }")

    return total_tof


def constraint_function(x):
    return x[1]**2 + x[2]**2


if __name__ == "__main__":
    x_a = np.arange(num_elements, dtype=np.float64) * pitch
    x_a = x_a - np.mean(x_a)
    z_a = np.ones_like(x_a) * d

    xf = np.linspace(x_a[0], x_a[-1], num_alpha_points)
    zf = np.ones((num_alpha_points,), dtype=np.float64) * d

    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)

    x_alpha, z_alpha = x_z_from_alpha(alpha)

    angle_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_points)
    x_pipe = r_outer * np.sin(angle_pipe)
    z_pipe = r_outer * np.cos(angle_pipe)
    
    bounds = Bounds([np.amin(alpha), -r_outer, np.amin(z_pipe), np.amin(alpha)], [np.amax(alpha), r_outer, np.amax(z_pipe), np.amax(alpha)])
    restr = NonlinearConstraint(constraint_function, r_outer**2 - 1e-15, r_outer**2 + 1e-15)
    
    initial_guess = np.asarray([alpha[num_alpha_points // 2 - 10], 0.0, r_outer, alpha[num_alpha_points // 2 + 10]])
    print(f"{initial_guess = }")

    results = []
    for emitter in range(20, 40, 1):
        print(f'Element shooting: {emitter}')

        for receiver in range(num_elements):
            print(f'Element target: {receiver}')

            fixed_args = (x_a[emitter], z_a[emitter], x_a[receiver], z_a[receiver], c1, c2)
            res = minimize(fermat, x0=initial_guess, args=fixed_args, bounds=bounds, constraints=[restr], tol=1e-18)

            if res["success"]:
                print(res)
                print(f"{emitter = }")
                print(f"{receiver = }")
                x_1, z_1 = x_z_from_alpha(res.x[0])
                x_p = res.x[1]
                z_p = res.x[2]
                x_2, z_2 = x_z_from_alpha(res.x[3])
                plot_setup(show=False, legend=False)
                plt.plot([x_a[emitter], x_1], [z_a[emitter], z_1], "C0", label="Incident ray")
                plt.plot([x_1, x_p], [z_1, z_p], "C1", label="Refracted ray (c1->c2)")
                plt.plot([x_p, x_2], [z_p, z_2], "C2", label="Reflected ray")
                plt.plot([x_2, x_a[receiver]], [z_2, z_a[receiver]], "C3", label="Refracted ray (c2->c1)")
                plt.legend()
                plt.show()

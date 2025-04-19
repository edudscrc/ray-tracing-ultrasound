import numpy as np
import matplotlib.pyplot as plt
import time

# Speeds of sound
c_lens = 6400
c_water = 1483
c_pipe = 5600

# Distances 'r' and 'z' for alpha = 0
r0 = 0.12156646438729327
z0 = 0.08843353561270673
d0 = r0 + z0

# Maximum sectorial angle (radians)
alpha_max = 50.62033040986099 * (np.pi / 180)
radius_pipe = 0.07

# Number of elements in the transducer
num_elements = 64
# Spacing between the elements
pitch = 0.0006

roi_angle_max = alpha_max * 0.9
roi_radius_max = radius_pipe
roi_radius_min = radius_pipe - 0.02


def roots_bhaskara(a, b, c):
    """Computes the roots of the polynomial ax^2 + bx + c = 0"""

    sqrt_delta = np.sqrt(b**2 - 4 * a * c)
    den = 2 * a
    x1 = (-b + sqrt_delta) / den
    x2 = (-b - sqrt_delta) / den

    return x1, x2


def z_r_from_alpha(alpha):
    """Computes the values of z and r of the lens for a given alpha"""

    t = r0 / c_lens + z0 / c_water
    a = c_lens**2 / c_water**2 - 1
    c = c_lens**2 * t**2 - d0**2

    phi2 = -2 * t * c_lens**2 / c_water
    phi3 = 2 * d0

    b = phi2 + phi3 * np.cos(alpha)

    z = roots_bhaskara(a, b, c)[1]
    r = np.sqrt(z**2 + d0**2 - 2 * z * d0 * np.cos(alpha))

    return z, r


def x_y_from_alpha(alpha):
    """Computes the (x,y) coordinates of the lens for a given alpha"""

    z, _ = z_r_from_alpha(alpha)
    y = z * np.cos(alpha)
    x = z * np.sin(alpha)

    return x, y


def dydx_from_alpha(alpha):
    """Computes the slope (dy/dx) of the lens for a given alpha"""

    t = r0 / c_lens + z0 / c_water

    def dzda_from_alpha(alpha):
        """Computes de derivative dz/dalpha for a given alpha"""

        # This function was originally created by Lucas Ocampos
        # at https://colab.research.google.com/drive/18yOqjGbSTO6MvzSMCYePM5cmVU8Ep0aa?usp=sharing

        a = c_lens**2 / c_water**2 - 1
        c = c_lens**2 * t**2 - d0**2

        phi1 = -1 / (2 * a)
        phi2 = -2 * t * c_lens**2 / c_water
        phi3 = 2 * d0

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
    z, _ = z_r_from_alpha(alpha)
    dydx = (dzda * np.sin(_alpha) + z * np.cos(_alpha)) / (dzda * np.cos(_alpha) - z * np.sin(_alpha))
    
    return dydx


def dxdy_pipe(x, r_pipe):
    """Computes the slope of the cilinder (pipe) for a given x"""
    return -x / np.sqrt(r_pipe**2 - x**2)


def rhp(x):
    """Projects an angle to the Right Half Plane [-pi/2; pi/2]"""
    x = np.mod(x, np.pi)
    x = x - (x > np.pi / 2) * np.pi
    x = x + (x < -np.pi / 2) * np.pi
    return x


def uhp(x):
    """Projects an angle to the Upper Half Plane [0; pi]"""
    x = rhp(x)
    x = x + (x < 0) * np.pi
    return x


def snell(gamma1, dydx, v1, v2):
    """Computes the the new angle  after the refraction
    of the first angle with the Snell's law (top-down)"""

    gamma1 = uhp(gamma1)
    slope = rhp(np.arctan(dydx))
    normal = slope + np.pi / 2
    theta1 = gamma1 - normal
    theta2 = np.arcsin(np.sin(theta1) * v2 / v1)
    gamma2 = slope - np.pi / 2 + theta2

    return gamma2


def distalpha(xc, yc, xf, yf, acurve):
    """For a shot fired from (xc, yc) in the direction x_y_from_alpha(alpha),
    this function computes the two refractions (from c1 to c2 and from
    c2 to c3) and then computes the squared distance between the ray at c3 and
    the "pixel" (xf, yf). A dictionary is returned with the following information:

    (x_lens, y_lens): position where the ray hits the lens
    (xcirc, ycirc): position where the ray hits the circle (cylinder)
    (xin, yin): point on the ray closest to (xf, yf)
    dist: squared distance (xin, yin) to (xf, yf)"""

    x_lens, y_lens = x_y_from_alpha(acurve)
    gamma1 = np.arctan((y_lens - yc) / (x_lens - xc))
    gamma1 = gamma1 + (gamma1 < 0) * np.pi
    dydx = dydx_from_alpha(acurve)
    gamma2 = snell(gamma1, dydx, c_lens, c_water)
    a_line = np.tan(uhp(gamma2))
    b_line = y_lens - a_line * x_lens
    a = a_line**2 + 1
    b = 2 * a_line * b_line
    c = b_line**2 - radius_pipe**2
    xcirc1, xcirc2 = roots_bhaskara(a, b, c)
    ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
    upper = ycirc1 > ycirc2
    xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
    ycirc = ycirc1 * upper + ycirc2 * (1 - upper)
    dxdycirc = dxdy_pipe(xcirc, radius_pipe)
    gamma3 = snell(gamma2, dxdycirc, c_water, c_pipe)
    a3 = np.tan(gamma3)
    b3 = ycirc - a3 * xcirc
    xbottom = -b3 / a3
    a4 = -1 / a3
    b4 = yf - a4 * xf
    xin = (b4 - b3) / (a3 - a4)
    yin = a3 * xin + b3
    dist = (xin - xf) ** 2 + (yin - yf) ** 2
    dic = {
        "x_lens": x_lens,
        "y_lens": y_lens,
        "xcirc": xcirc,
        "ycirc": ycirc,
        "xin": xin,
        "yin": yin,
        "dist": dist,
    }
    return dic


def dist_and_derivatives(xc, yc, xf, yf, alpha_lens, eps=1e-5):
    """Computes the squared distance using distalpha as well as the first and
    second derivatives of the squared distance with relation to alpha."""
    dm = distalpha(xc, yc, xf, yf, alpha_lens - eps)["dist"]
    dic = distalpha(xc, yc, xf, yf, alpha_lens)
    d0 = dic["dist"]
    dp = distalpha(xc, yc, xf, yf, alpha_lens + eps)["dist"]
    der1 = (dp - dm) * 0.5 / eps
    der2 = (dm - 2 * d0 + dp) / eps**2
    return dic, der1, der2


def newton(xc, yc, xf, yf, alpha_initial_guess=None, iter=6):
    """Uses the Newton-Raphson method.

    Computes the direction in which the transducer at (xc, yc) should fire in order to hit the "pixel" at (xf, yf).

    Parameters
    ----------
    xc : float
    	x-coordinate of the transducer's element.
    yc : float
    	y-coordinate of the transducer's element.
    xf : ndarray
    	Array with x-coordinate of each target.
    yf : ndarray
    	Array with y-coordinate of each target.
    alpha_initial_guess : ndarray
    	Array with alpha values.
    iter : int
    	Number of iterations for Newton-Raphson method.

    Returns
    -------
    dic : dict
		(x_lens, y_lens): position where the ray hits the lens.
		(xcirc, ycirc): position where the ray hits the circle (cylinder).
		(xin, yin): point on the ray closest to (xf, yf).
		dist: squared distance (xin, yin) to (xf, yf) (should be close to zero).
		maxdist: maximum squared distance (assuming an array of pixels was passed).
		maxdist: maximum squared distance (assuming an array of pixels was passed).
    """

    if alpha_initial_guess is None:
        alpha_initial_guess = np.arctan(xf / yf)

    maxdist = []
    mindist = []
    for _ in range(iter):
        dic, d1, _ = dist_and_derivatives(xc, yc, xf, yf, alpha_initial_guess, eps=1e-4)
        alpha_initial_guess -= dic["dist"] / d1
        alpha_initial_guess[alpha_initial_guess > alpha_max] = alpha_max * 0.9
        alpha_initial_guess[alpha_initial_guess < -alpha_max] = -alpha_max * 0.9
        maxdist.append(dic["dist"].max())
        mindist.append(dic["dist"].min())
    dic["maxdist"] = maxdist
    dic["mindist"] = mindist
    return dic


def newton_batch(xc, yc, xf, yf, iter=6):
    """Calls the function newton() one time for each transducer element.

        The set of angles found for a given element are used as initial guess for the next one.
        Starts from the center of the transducer.

        Parameters
        ----------
        xc : ndarray
			Array with x-coordinate of each transducer's element.
        yc : ndarray
			Array with y-coordinate of each transducer's element.
        xf : ndarray
			Array with x-coordinate of each target.
        yf : ndarray
			Array with y-coordinate of each target.
        iter : int
			Number of iterations for Newton-Raphson method.

        Returns
        -------
        results : dict
			(x_lens, y_lens): position where the ray hits the lens.
			(xcirc, ycirc): position where the ray hits the circle (cylinder).
			(xin, yin): point on the ray closest to (xf, yf).
			dist: squared distance (xin, yin) to (xf, yf) (should be close to zero).
			maxdist: maximum squared distance (assuming an array of pixels was passed).
        """

    transducer_center = num_elements // 2

    results = [None] * num_elements

    results[transducer_center] = newton(
        xc=xc[transducer_center],
        yc=yc[transducer_center],
        xf=xf,
        yf=yf,
        iter=iter
    )
    results[transducer_center - 1] = newton(
        xc=xc[transducer_center - 1],
        yc=yc[transducer_center - 1],
        xf=xf,
        yf=yf,
        iter=iter
    )

    for i in range(1, transducer_center):
        i_minus = transducer_center - i - 1
        i_plus = transducer_center + i

		# Newton-Raphson method for i_plus
        alpha_initial_guess_plus = np.arctan(results[i_plus - 1]["x_lens"] / results[i_plus - 1]["y_lens"])
        results[i_plus] = newton(
            xc=xc[i_plus],
            yc=yc[i_plus],
            xf=xf,
            yf=yf,
            alpha_initial_guess=alpha_initial_guess_plus,
            iter=iter
        )
        bad_indices = results[i_plus]["dist"] > 1e-8
        num_bad_indices = np.count_nonzero(bad_indices)
        if num_bad_indices > 0:
            print(f"Bad indices found at {i_plus}: {num_bad_indices}")

		# Newton-Raphson method for i_minus
        alpha_initial_guess_minus = np.arctan(results[i_minus + 1]["x_lens"] / results[i_minus + 1]["y_lens"])
        results[i_minus] = newton(
            xc=xc[i_minus],
            yc=yc[i_minus],
            xf=xf,
            yf=yf,
            alpha_initial_guess=alpha_initial_guess_minus,
            iter=iter
        )
        bad_indices = results[i_minus]["dist"] > 1e-8
        num_bad_indices = np.count_nonzero(bad_indices)
        if num_bad_indices > 0:
            print(f"Bad indices found at {i_plus}: {num_bad_indices}")

        print(f"Computed {i * 2} elements (of {num_elements})")

    return results


def plot_diamond():
    """Plots the main features of the setup."""

    num_alpha_points = 101
    alpha = np.linspace(-alpha_max, alpha_max, num_alpha_points)
    x_lens, y_lens = x_y_from_alpha(alpha)

    num_alpha_pipe_points = 201
    alpha_pipe = np.linspace(-np.pi / 2, np.pi / 2, num_alpha_pipe_points)
    x_pipe = radius_pipe * np.sin(alpha_pipe)
    y_pipe = radius_pipe * np.cos(alpha_pipe)

    plt.figure()
    plt.plot(x_lens, y_lens, "-k")
    plt.plot(x_pipe, y_pipe, "-C0")
    plt.plot([0], [0], "or")
    plt.plot([0], [d0], "sk")
    plt.axis("equal")
    plt.show()


########## COMPUTATION OF ENTRY POINTS WITH NEWTON'S METHOD ########





# i_elem = 34

# plt.figure()
# plt.semilogy(results[i_elem]["maxdist"], "o-")
# plt.semilogy(results[i_elem]["mindist"], "o-")
# plt.grid()
# plt.xlabel("iteration")
# plt.ylabel("Distances")
# plt.legend(["Max distance", "Min distance"])
# plt.title("Newton algorithm convergence fof element " + str(i_elem))


# def dist(x1, y1, x2, y2):
#     return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# n = 1
# tof = dist(xc[n], yc[n], results[n]["xcurve"], results[n]["ycurve"]) / c1
# tof += (
#     dist(
#         results[n]["xcurve"],
#         results[n]["ycurve"],
#         results[n]["xcirc"],
#         results[n]["ycirc"],
#     )
#     / c2
# )
# tof += dist(results[n]["xcirc"], results[n]["ycirc"], xf, yf) / c3
# tof = tof.reshape((len(rf), len(af)))
# plt.figure()
# plt.imshow(tof)
# plt.colorbar()
# plt.axis("auto")
# plt.title("Times of flight for element " + str(N_elem - 1))


# plot_diamond()
# plt.plot(xc, yc, ".k")
# for i in np.arange(0, len(af), 10):
#     plt.plot(
#         [xc[n], results[n]["xcurve"][i], results[n]["xcirc"][i], xf[i]],
#         [yc[n], results[n]["ycurve"][i], results[n]["ycirc"][i], yf[i]],
#         "C2",
#         alpha=0.3,
#     )

if __name__ == "__main__":
    plot_diamond()

        # 'xc' and 'yc' are arrays of the positions (x, y) of each transducer's element
    xc = np.arange(num_elements) * pitch
    xc = xc - np.mean(xc)
    yc = np.ones_like(xc) * d0

    num_roi_angle_points = 181
    af = np.linspace(-roi_angle_max, roi_angle_max, num_roi_angle_points)
    rf = np.linspace(roi_radius_min, roi_radius_max, 1)
    Af, Rf = np.meshgrid(af, rf)
    Af = Af.flatten()
    Rf = Rf.flatten()

        # 'xf' and 'yf' are arrays of the positions (x, y) of each target (the suffix 'f' means fire)
    xf = Rf * np.sin(Af)
    yf = Rf * np.cos(Af)

    start = time.time()
    results = newton_batch(xc, yc, xf, yf, iter=20)
    end = time.time()
    print(f"Elapsed time - newton_batch: {end - start} seconds.")

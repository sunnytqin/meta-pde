import os
import jax.numpy as np
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
vmap_polyval = vmap(np.polyval, (0, None), -1)
import jax
from jax.config import config
config.update("jax_enable_x64", True)

from sympy import legendre, diff, integrate, symbols
from functools import lru_cache, partial
import numpy as onp
import matplotlib.pyplot as plt
import time
from absl import app
from absl import flags
from ..util import common_flags

from .td_burgers_fenics import solve_fenics


FLAGS = flags.FLAGS


PI = np.pi


def upper_B(m, k):
    x = symbols("x")
    expr = x ** k * (x + 0.5) ** m
    return integrate(expr, (x, -1, 0))


def lower_B(m, k):
    x = symbols("x")
    expr = x ** k * (x - 0.5) ** m
    return integrate(expr, (x, 0, 1))


def A(m, k):
    x = symbols("x")
    expr = legendre(k, x) * x ** m
    return integrate(expr, (x, -1, 1)) / (2 ** (m + 1))


@lru_cache(maxsize=10)
def get_B_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(2 * p):
            res[m, k] = upper_B(m, k)
    for m in range(p):
        for k in range(2 * p):
            res[m + p, k] = lower_B(m, k)
    return res


@lru_cache(maxsize=10)
def get_inverse_B(p):
    B = get_B_matrix(p)
    return onp.linalg.inv(B)


@lru_cache(maxsize=10)
def get_A_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(p):
            res[m, k] = A(m, k)
            res[m + p, k + p] = A(m, k)
    return res


def get_b_coefficients(a):
    p = a.shape[1]
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    b = np.einsum("km,jm->jk", B_inv, rhs)
    return b


def recovery_slope(a, p):
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)[1, :]
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    slope = np.einsum("m,jm->j", B_inv, rhs)
    return slope

def add_ghost_cells(a):
    return np.concatenate([-a[0:1], a, -a[-1:]])


def enforce_ghost_cells(a):
    a = a.at[0].set(-a[1])
    return a.at[-1].set(-a[-2])

def ssp_rk3(a_n, t_n, F, dt):
    a_1 = enforce_ghost_cells(a_n + dt * F(a_n, t_n))
    a_2 = enforce_ghost_cells(0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt)))
    a_3 = enforce_ghost_cells(1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2)))
    return a_3, t_n + dt


def _quad_two_per_interval(f, a, b, n=5):
    mid = (a + b) / 2
    return _fixed_quad(f, a, mid, n) + _fixed_quad(f, mid, b, n)


def _fixed_quad(f, a, b, n=5):
    assert isinstance(n, int) and n <= 8 and n > 0
    w = {
        1: np.asarray([2.0]),
        2: np.asarray([1.0, 1.0]),
        3: np.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: np.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: np.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: np.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: np.asarray(
            [
                0.1294849661688696932706,
                0.2797053914892766679015,
                0.38183005050511894495,
                0.417959183673469387755,
                0.38183005050511894495,
                0.279705391489276667901,
                0.129484966168869693271,
            ]
        ),
        8: np.asarray(
            [
                0.1012285362903762591525,
                0.2223810344533744705444,
                0.313706645877887287338,
                0.3626837833783619829652,
                0.3626837833783619829652,
                0.313706645877887287338,
                0.222381034453374470544,
                0.1012285362903762591525,
            ]
        ),
    }[n]

    xi_i = {
        1: np.asarray([0.0]),
        2: np.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: np.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: np.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: np.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: np.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: np.asarray(
            [
                -0.9491079123427585245262,
                -0.7415311855993944398639,
                -0.4058451513773971669066,
                0.0,
                0.4058451513773971669066,
                0.7415311855993944398639,
                0.9491079123427585245262,
            ]
        ),
        8: np.asarray(
            [
                -0.9602898564975362316836,
                -0.7966664774136267395916,
                -0.5255324099163289858177,
                -0.1834346424956498049395,
                0.1834346424956498049395,
                0.5255324099163289858177,
                0.7966664774136267395916,
                0.9602898564975362316836,
            ]
        ),
    }[n]

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return np.sum(wprime[:, None] * f(x_i), axis=0)


def inner_prod_with_legendre(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    
    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    ) 
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(np.asarray([[1.]]), xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)





def map_f_to_FV(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    return (
        inner_prod_with_legendre(f, t, nx, dx, quad_func=quad_func, n=n) / dx
    )



def evalf_1D(x, a, dx, leg_poly):
    j = np.floor(x / dx).astype(int)
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    poly_eval = vmap_polyval(np.asarray([[1.]]), xi)  # nx, p array
    return np.sum(poly_eval * a[j, :], axis=-1)


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a_f

def _godunov_flux_1D_burgers(a):
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(a[1:], axis=-1)
    zero_out = 0.5 * np.abs(np.sign(u_left) + np.sign(u_right))
    compare = np.less(u_left, u_right)
    F = lambda u: u ** 2 / 2
    return compare * zero_out * np.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * np.maximum(F(u_left), F(u_right))

def _diffusion_term_1D_burgers(a, t, dx, nu):
    negonetok = (np.ones(1) * -1) ** np.arange(1)
    slope_right = recovery_slope(a, 1) / dx
    slope_left = np.roll(slope_right, 1)
    return nu * (slope_right[:, None] - negonetok[None, :] * slope_left[:, None])


def time_derivative_1D_burgers(
    a, t, dx, flux, nu
):
    if flux == "godunov":
        flux_right = _godunov_flux_1D_burgers(a)
    else:
        raise Exception

    if nu > 0.0:
        dif_term = _diffusion_term_1D_burgers(a, t, dx, nu)
    else:
        dif_term = 0.0

    flux_left = np.roll(flux_right, 1, axis=0)
    flux_term = (flux_left[:, None] - flux_right[:, None])
    return (flux_term + dif_term) / dx
    

def simulate_1D(
    a0,
    t0,
    dx,
    dt,
    nt,
    nu = 0.0,
    output=False,
    rk=ssp_rk3,
    flux="godunov"
):

    dadt = lambda a, t: time_derivative_1D_burgers(
        a,
        t,
        dx,
        flux,
        nu,
    )

    rk_F = lambda a, t: rk(a, t, dadt, dt)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return (a_f, t_f)


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    ##### for the purposes of debugging
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(np.asarray([[1.]]), xi)
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    ys = vmap_eval(xs, a, np.arange(nx), dx)
    subfig.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return xs, ys

def print_runtime(func):
    #### decorator that prints the simulation time
    def wrapper(*args, **kwargs):
        ti = time.time()
        output = func(*args, **kwargs)
        tf = time.time()
        print("time to simulate is {} microseconds".format(int(10**6 * (tf - ti))))
        return output
    return wrapper

###################################################################################################################
###################################################################################################################
# IGNORE ABOVE THIS LINE
###################################################################################################################
###################################################################################################################

#######################
# Runtime Hyperparameters
#######################

def main(argv):
    t0 = 0.0
    Tf = 1.0
    L = 1.0
    nxs = [16, 32, 64, 128, 256]
    nu = 1/100
    cfl_safety_factor = 0.8
    MAX = 2.0
    num_plot = 2

    T = Tf/(num_plot-1)

    key = jax.random.PRNGKey(FLAGS.seed)

    ###################################################################################################################
    ###################################################################################################################
    # PART 1: RUNTIME DEMO
    ###################################################################################################################
    ###################################################################################################################


    ##################
    # Generate initial conditions
    ##################
    k1, k2 = jax.random.split(key, 2)
    ic_params = jax.random.uniform(k2, shape=(2,), minval=-MAX, maxval=MAX)
    f_init = lambda x, t: np.sin(PI * x) + ic_params[0] * np.sin(2 * PI * x) + ic_params[1] * np.sin(4 * PI * x)
    ##################
    # End Generate Initial Conditions
    ##################



    #######################
    # Begin Simulate
    #######################

    for nx in nxs:
        dx = L/nx

        a0 = map_f_to_FV(f_init, t0, nx, dx) # (nx, 1) array
        print("nx is {}".format(a0.shape[0]))

        @partial(jax.jit, static_argnums=(2,))
        def sim(a0, t0, T):
            nt = int(T / (cfl_safety_factor * dx / (1 + MAX + MAX))) + 1
            dt = T / nt
            a0 = add_ghost_cells(a0)
            af, tf = simulate_1D(a0, t0, dx, dt, nt, nu)
            return af[1:-1], tf

        sim(a0, t0, T) # jit-compile first
        (af, tf) = print_runtime(sim)(a0, t0, T)


        ##################
        # End Simulate, Begin Plot
        ##################

        stuff_to_plot = [a0, af]

        fig, axs = plt.subplots(1, num_plot, sharex=True, sharey=True, squeeze=True, figsize=(8,8/4))
        for j in range(num_plot):
            xs, ys = plot_subfig(stuff_to_plot[j], axs[j], L, color="#ff5555", linewidth=1.0)
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([-4.0, 4.0])





    ###################################################################################################################
    ###################################################################################################################
    # PART 2: Compute losses as a function of time
    ###################################################################################################################
    ###################################################################################################################


    #######################
    # Loss Hyperparameters
    #######################

    key = jax.random.PRNGKey(FLAGS.seed)
    k1, _ = jax.random.split(key)
    t0 = 0.0
    Tf = 1.0
    L = 1.0
    nu = 1/100
    cfl_safety_factor = 0.5
    cfl_safety_factor_exact = 0.1
    MAX = 2.0
    dx = L/nx

    num_loss = 16

    nx_base = 16
    num_upsampling = 5
    upsamplings = 2**np.arange(num_upsampling)
    exact_upsampling = 2**num_upsampling


    ############################
    # Compute losses
    ############################


    nx_exact = nx_base * exact_upsampling
    dx_exact = L / nx_exact
    nt_exact = int(T / (cfl_safety_factor_exact * dx_exact / (1 + MAX + MAX)))
    dt_exact = T / nt_exact
    assert nx_exact < 600

    print("Values of nx are {}, nx_exact is {}, nt_exact is {}".format(nx_base * upsamplings, nx_exact, nt_exact))

    assert (nt_exact % exact_upsampling == 0)

    def loss(a1, a2):
        return np.mean((a1 - a2)**2)

    def compute_a(f_init, upsampling):
        nx = nx_base * upsampling
        dx = L / nx
        a0 = add_ghost_cells(map_f_to_FV(f_init, t0, nx, dx))
        dt = dt_exact / (upsampling / exact_upsampling) * (cfl_safety_factor / cfl_safety_factor_exact)
        nt = nt_exact * (upsampling / exact_upsampling) / (cfl_safety_factor / cfl_safety_factor_exact)
        return simulate_1D(a0, t0, dx, dt, nt, nu, output=True)[:, 1:-1]

    def compute_a_exact(f_init):
        a0_exact = add_ghost_cells(map_f_to_FV(f_init, t0, nx_exact, dx_exact))
        return simulate_1D(a0_exact, t0, dx_exact, dt_exact, nt_exact, nu, output=True)[:,1:-1]

    def downsample(a_exact, upsampling):
        UP = int(exact_upsampling / upsampling)
        T_UP = int(UP * (cfl_safety_factor / cfl_safety_factor_exact))
        a_exact_time_downsampled = a_exact[::T_UP]
        nt, nx_exact, _ = a_exact_time_downsampled.shape
        return np.mean(a_exact_time_downsampled.reshape(nt, -1, UP, 1), axis=2)

    losses = onp.zeros(num_upsampling)
    solving_time = onp.zeros(num_upsampling)
    for _ in range(num_loss):
        k1, k2 = jax.random.split(k1, 2)
        ic_params = jax.random.uniform(k2, shape=(2,), minval=-MAX, maxval=MAX)
        f_init = lambda x, t: np.sin(PI * x) + ic_params[0] * np.sin(2 * PI * x) + ic_params[1] * np.sin(4 * PI * x)
        tic = time.time()
        a_exact = compute_a_exact(f_init)
        toc = time.time()

        #a_exact_downsampled = downsample(a_exact, 2)

        #nt = a_exact_downsampled.shape[0]
        #dt = T / nt
        #tjs = np.arange(nt) * dt
        #print("tjs shape:", tjs[0:10])

        #nx = a_exact_downsampled.shape[1]
        #dx = L / nx
        #xjs = (np.arange(nx) * dx)
        #print("xjs shape:", xjs.shape)

        # also compute fenics baseline
        #params = (np.array([1./nu]), ic_params)
        #ground_truth = solve_fenics(params, boundary_points=128, resolution=512)
        #ground_truth.set_allow_extrapolation(True)
        #gt_fenics = onp.empty_like(a_exact_downsampled)
        #for i in range(nt):
        #    for j in range(nx):
        #        gt_fenics[i, j] = ground_truth((xjs[j], tjs[i]))
        #ground_truth.set_allow_extrapolation(False)
        #print("gt_fenics: ", loss(gt_fenics, a_exact_downsampled))
        for j, upsampling in enumerate(upsamplings):
            tic = time.time()
            a = compute_a(f_init, upsampling)
            toc = time.time()
            print("a shape: ", a.shape)
            a_exact_downsampled = downsample(a_exact, upsampling)
            losses[j] += loss(a, a_exact_downsampled) / num_loss
            solving_time[j] += (toc - tic) / num_loss

    fig2, axs2 = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=True, figsize=(8,8/4))

    axs2.loglog(upsamplings, losses)
    for t, l in zip(solving_time, losses):
        print(f"solving time {t} with loss {l}")

    path = 'td_burgers_fenics_results/jax'
    onp.save(os.path.join(path, "errors_by_resolution.npy"), (solving_time, losses), allow_pickle=True)

    #plt.show()


if __name__ == "__main__":
    app.run(main)

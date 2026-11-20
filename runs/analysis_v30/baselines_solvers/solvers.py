"""Item 5 (v30): inversion / regularisation methods for the joint model s = A g, all matrix-free (A and A^H are
sata1d.joint_line.LineModel.forward / adjoint; nothing N_az x N_az is formed).

Methods (each returns g [N_az, lag order] and an info dict with the parameter, the number of A and A^H calls, time):
  current      the repository's solver: Tikhonov, mu = 0.03 sigma_max(A), CGLS <= 60 iterations / tol 1e-4, 2 passes
               (iterated Tikhonov) - identical to reconstruct_joint_line (checked in check_baseline()).
  lcurve       Tikhonov, mu at the corner of the L-curve (log ||A g - s||, log ||g||), found with the RelSea search region
               + Menger curvature / golden-section search (Gonzalez-Caboverde et al. 2026, Algorithms 1 and 3; "EMA").
  discrepancy  Tikhonov, mu from Morozov's discrepancy principle ||A g - s||^2 = M N0 (needs the noise power N0).
  bmr          statistical (Bayes minimum risk) regularisation with the delta-correlated model R_s = S0 I, R_n = N0 I:
               F = (A^H A + N0/S0 I)^-1 A^H (Martin del Campo et al. 2020, eq. 59), S0 from the data:
               S0 = (||s||^2 - M N0) / tr(A^H A), tr(A^H A) by Hutchinson probes.
  maria        ML-inspired adaptive robust iterative approach (Martin del Campo et al. 2018, Table II of the 2020 paper),
               single look (Y = s s^H): b_m <- P_gamma{ |a_m^H R^-1 s|^2 / (a_m^H R^-1 a_m) b_m },
               R = A D(b) A^H + N0 I; complex estimate g = D(b) A^H R^-1 s.
  wise         WCF-based iterative spectral estimator (2020 paper, Table III): b_m <- P_gamma{ tr(Y) |a_m^H R^-1 s|^2 /
               (a_m^H a_m) b_m }.
For maria / wise, R^-1 s is applied through g = W u, u = argmin ||A W u - s||^2 + N0 ||u||^2, W = D(b)^(1/2) (CGLS on
A W), and the diagonal a_m^H R^-1 a_m is taken in the matched-filter approximation d / (d b_m + N0), d = a_m^H a_m
(exact when A^H A is diagonal; A^H A has the same diagonal d for every place because the DEM screens have unit modulus).
The zero-step estimate is the matched spatial filter b0 = |A^H s|^2 / d^2 (the single-look stand-in for Capon, which
needs many looks), as in the 2018 letter. N0 is the noise power if known (noisy runs) and otherwise a diagonal loading
equal to the current method's (0.03 sigma_max)^2 times mean(b0).
"""
import time

import numpy as np


class Op:
    """A (forward) and A^H (adjoint) of a LineModel, with call counters; optional column weights w (A diag(w))."""

    def __init__(self, model, w=None):
        self.M = model
        self.w = w
        self.na = 0
        self.nah = 0

    def A(self, g):
        self.na += 1
        return self.M.forward(g if self.w is None else self.w * g)

    def AH(self, s):
        self.nah += 1
        x = self.M.adjoint(s)
        return x if self.w is None else np.conj(self.w) * x


def cgls(op, r, damp2, iterations=60, tol=1e-4):
    """min ||A d - r||^2 + damp2 ||d||^2 (same iteration as reconstruct_joint_line). Returns d, residual r - A d, its."""
    d = np.zeros(op.M.cfg.Na, complex)
    rp = r.copy()
    z = op.AH(rp)
    p = z.copy()
    zz = z0 = np.vdot(z, z).real
    n = 0
    for _ in range(iterations):
        q = op.A(p)
        alpha = zz / (np.vdot(q, q).real + damp2 * np.vdot(p, p).real)
        d += alpha * p
        rp -= alpha * q
        n += 1
        z = op.AH(rp) - damp2 * d
        zz_new = np.vdot(z, z).real
        if zz_new <= tol ** 2 * z0:
            break
        p = z + (zz_new / zz) * p
        zz = zz_new
    return d, rp, n


def sigma_max(op, iterations=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(op.M.cfg.Na) + 1j * rng.standard_normal(op.M.cfg.Na)
    n = 1.0
    for _ in range(iterations):
        x = op.AH(op.A(x))
        n = np.linalg.norm(x)
        x /= n
    return float(np.sqrt(n))


def tikhonov(op, s, mu, passes=1, iterations=60, tol=1e-4):
    g = np.zeros(op.M.cfg.Na, complex)
    r = s.copy()
    its = 0
    for _ in range(passes):
        d, r, n = cgls(op, r, mu ** 2, iterations, tol)
        g += d
        its += n
    return g, r, its


def _info(op, t0, **kw):
    kw.update(calls_A=op.na, calls_AH=op.nah, sec=time.time() - t0)
    return kw


# ------------------------------------------------------------------ methods
def current(model, s, reg=0.03, passes=2):
    op = Op(model)
    t0 = time.time()
    sm = sigma_max(op)
    g, r, its = tikhonov(op, s, reg * sm, passes=passes)
    return g, _info(op, t0, mu=reg * sm, mu_rel=reg, sigma_max=sm, iterations=its,
                    resid=float(np.linalg.norm(r) / np.linalg.norm(s)))


def _menger(P1, P2, P3):
    """Menger curvature of three L-curve points (tau, eta) (RelSea eq. 13); > 0 = bends like the corner of an L."""
    (x1, y1), (x2, y2), (x3, y3) = P1, P2, P3
    num = 2.0 * (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2)
    den = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * ((x1 - x3) ** 2 + (y1 - y3) ** 2))
    return num / den if den > 0 else 0.0


def lcurve(model, s, n_grid=9, eps=0.02, passes=1, lo_rel=1e-4):
    """L-curve corner. Step 1 (RelSea search region): the spectral interval [lo_rel sigma_max, sigma_max] on a coarse
    log grid; keep the convex part (positive Menger curvature) around the point of largest curvature. Step 2 (EMA):
    Menger curvature + golden-section search (RelSea Algorithm 1 with the interval update of Algorithm 3) inside it."""
    op = Op(model)
    t0 = time.time()
    sm = sigma_max(op)
    cache = {}

    def L(a):
        k = float(np.round(np.log10(a), 6))
        if k not in cache:
            g, r, _ = tikhonov(op, s, a, passes=passes)
            cache[k] = (np.log(np.linalg.norm(r) ** 2), np.log(np.linalg.norm(g) ** 2), g, r)
        return cache[k][:2]

    lg = np.linspace(np.log10(lo_rel * sm), np.log10(sm), n_grid)
    pts = [L(10 ** b) for b in lg]
    curv = [_menger(pts[i - 1], pts[i], pts[i + 1]) for i in range(1, n_grid - 1)]
    i_max = int(np.argmax(curv)) + 1
    a_lo, a_hi = i_max - 1, i_max + 1                  # widen while the curvature stays positive (convex region)
    while a_lo > 0 and curv[a_lo - 1] > 0:
        a_lo -= 1
    while a_hi < n_grid - 1 and curv[a_hi - 1] > 0:
        a_hi += 1
    b1, b4 = lg[a_lo], lg[a_hi]
    region = (10 ** b1, 10 ** b4)
    phi = (1 + 5 ** 0.5) / 2
    b2 = (b4 + phi * b1) / (1 + phi)
    b3 = b1 + (b4 - b2)
    corner = 10 ** lg[i_max]
    it = 0
    while (b4 - b1) > eps and it < 30:                 # Algorithm 1 (MC + golden section), on log10 alpha
        it += 1
        C3 = _menger(L(10 ** b2), L(10 ** b3), L(10 ** b4))
        guard = 0
        while C3 < 0 and guard < 10:
            guard += 1
            b4, b3 = b3, b2
            b2 = (b4 + phi * b1) / (1 + phi)
            C3 = _menger(L(10 ** b2), L(10 ** b3), L(10 ** b4))
        C2 = _menger(L(10 ** b1), L(10 ** b2), L(10 ** b3))
        if C2 > C3:
            corner = 10 ** b2
            b4, b3 = b3, b2
            b2 = (b4 + phi * b1) / (1 + phi)
        else:
            corner = 10 ** b3
            b1, b2 = b2, b3
            b3 = b1 + (b4 - b2)
    g, r, _ = tikhonov(op, s, corner, passes=passes)
    curve = sorted((10 ** k, v[0], v[1]) for k, v in cache.items())
    return g, _info(op, t0, mu=corner, mu_rel=corner / sm, sigma_max=sm, region_rel=(region[0] / sm, region[1] / sm),
                    golden_iterations=it, n_solves=len(cache) + 1, curve=curve,
                    resid=float(np.linalg.norm(r) / np.linalg.norm(s)))


def discrepancy(model, s, N0, tau=1.0, passes=1, lo_rel=1e-5):
    """Morozov: the largest mu with ||A g_mu - s||^2 <= tau M N0 (bisection on log mu)."""
    op = Op(model)
    t0 = time.time()
    sm = sigma_max(op)
    target = tau * s.size * N0
    lo, hi = np.log10(lo_rel * sm), np.log10(sm)
    best = None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        g, r, _ = tikhonov(op, s, 10 ** mid, passes=passes)
        if np.linalg.norm(r) ** 2 <= target:
            lo, best = mid, (g, r, 10 ** mid)
        else:
            hi = mid
    if best is None:
        g, r, _ = tikhonov(op, s, 10 ** lo, passes=passes)
        best = (g, r, 10 ** lo)
    g, r, mu = best
    return g, _info(op, t0, mu=mu, mu_rel=mu / sm, sigma_max=sm, resid=float(np.linalg.norm(r) / np.linalg.norm(s)))


def trace_AHA(op, probes=16, seed=1):
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(probes):
        z = (rng.standard_normal(op.M.cfg.Na) + 1j * rng.standard_normal(op.M.cfg.Na)) / np.sqrt(2)
        acc += np.linalg.norm(op.A(z)) ** 2
    return acc / probes


def bmr(model, s, N0, passes=1):
    """F = (A^H A + N0/S0 I)^-1 A^H: Tikhonov with mu^2 = N0 / S0 (noise power over mean scene power)."""
    op = Op(model)
    t0 = time.time()
    sm = sigma_max(op)
    trA = trace_AHA(op)
    S0 = max((np.linalg.norm(s) ** 2 - s.size * N0) / trA, 1e-30)
    mu = np.sqrt(N0 / S0)
    g, r, its = tikhonov(op, s, mu, passes=passes)
    return g, _info(op, t0, mu=mu, mu_rel=mu / sm, sigma_max=sm, S0=S0, N0=N0, iterations=its,
                    resid=float(np.linalg.norm(r) / np.linalg.norm(s)))


def column_norm2(model):
    """d = a_m^H a_m (the same for every place m: unit-modulus screens, circulant kernels decimated by N_rx)."""
    e = np.zeros(model.cfg.Na, complex)
    e[0] = 1.0
    return float(np.linalg.norm(model.forward(e)) ** 2)


def adaptive(model, s, kind="maria", N0=None, n_iter=10, tol=1e-3, gamma=0.0, reg=0.03, init="msf", g_init=None,
             diag="mf", probes=16):
    """MARIA / WISE, single look, matrix-free (see the module docstring)."""
    op = Op(model)
    t0 = time.time()
    sm = sigma_max(op)
    d = column_norm2(model)
    if init == "msf":
        b = np.abs(op.AH(s)) ** 2 / d ** 2
    else:
        b = np.abs(g_init) ** 2
    if N0 is None:
        N0 = (reg * sm) ** 2 * float(np.mean(b))
        n0_kind = "diagonal loading (0.03 sigma_max)^2 mean(b0)"
    else:
        n0_kind = "noise power"
    trY = np.linalg.norm(s) ** 2
    hist = []
    g = None
    for i in range(n_iter):
        w = np.sqrt(b)
        opw = Op(model, w)
        u, r, _ = cgls(opw, s, N0)
        op.na += opw.na
        op.nah += opw.nah
        g = w * u                                      # = D(b) A^H R^-1 s, so a_m^H R^-1 s = u_m / sqrt(b_m)
        qb = np.abs(u) ** 2                            # |a_m^H R^-1 s|^2 b_m (no division by b)
        if kind == "maria":
            if diag == "mf":
                bn = qb * (d * b + N0) / d             # / (a_m^H R^-1 a_m), matched-filter approximation d/(d b + N0)
            else:
                bn = qb / np.maximum(diag_ARA(model, b, N0, op, probes=probes), 1e-30)
        else:
            bn = trY * qb / d
        bn[bn < gamma] = 0.0                           # POCS projector P_gamma
        ch = np.linalg.norm(bn - b) / max(np.linalg.norm(b), 1e-30)
        hist.append(float(ch))
        b = bn
        if ch < tol:
            break
    return g, _info(op, t0, N0=N0, N0_kind=n0_kind, sigma_max=sm, d=d, outer_iterations=len(hist), change=hist,
                    mu=float(np.sqrt(N0 / max(np.mean(b), 1e-30))), mu_rel=float(np.sqrt(N0 / max(np.mean(b), 1e-30)) / sm))


def diag_ARA(model, b, N0, op, probes=16, seed=2, iterations=200, tol=1e-4):
    """Hutchinson estimate of diag(A^H R^-1 A), R = A D(b) A^H + N0 I (CG on R for every probe)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(model.cfg.Na)

    def R(x):
        return op.A(b * op.AH(x)) + N0 * x

    for _ in range(probes):
        z = rng.choice([-1.0, 1.0], model.cfg.Na) + 0j
        y = op.A(z)
        x = np.zeros_like(y)
        rr = y.copy()
        p = rr.copy()
        rs = np.vdot(rr, rr).real
        r0 = rs
        for _ in range(iterations):
            Rp = R(p)
            a = rs / np.vdot(p, Rp).real
            x += a * p
            rr -= a * Rp
            rn = np.vdot(rr, rr).real
            if rn <= tol ** 2 * r0:
                break
            p = rr + (rn / rs) * p
            rs = rn
        acc += np.real(np.conj(z) * op.AH(x))
    return acc / probes


def to_line(model, g, zero_out=True):
    """As reconstruct_joint_line: back to the raw line of the scene at the full PRF, limited to B_a."""
    cfg = model.cfg
    U = np.fft.fft(g) * model.chirp
    if zero_out:
        U[np.abs(np.fft.fftfreq(cfg.Na, 1.0 / cfg.prf)) > cfg.abw / 2] = 0
    return np.fft.ifft(U)

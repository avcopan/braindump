import warnings

import numpy as np
import scipy.linalg as spla


def density(c, slc=slice(None)):
    return np.dot(c[:, slc], c[:, slc].T)


def projection(s, d):
    return np.dot(d, s)


def projection_complement(s, d):
    n = s.shape[0]
    assert s.shape == d.shape == (n, n)
    return np.eye(n) - np.dot(d, s)


def coulomb(g, d):
    return np.tensordot(g, d, axes=[(1, 3), (0, 1)])


def exchange(g, d):
    return np.tensordot(g, d, axes=[(1, 2), (0, 1)])


def mean_field(g, ad, bd):
    j = coulomb(g, ad + bd)
    ak = exchange(g, ad)
    bk = exchange(g, bd)
    return j - ak, j - bk


def fock(h, g, ad, bd):
    aw, bw = mean_field(g, ad, bd)
    return h + aw, h + bw


def fock_block(f, p_row, p_col):
    return np.linalg.multi_dot([p_row.T, f, p_col])


def fock_block_sum(f, p_pairs, weights=None):
    if weights is None:
        weights = np.ones(len(p_pairs))
    return np.sum(c * fock_block(f, p_row, p_col)
                  for (p_row, p_col), c in zip(p_pairs, weights))


def rohf_fock(s, af, bf, ad, bd):
    n = s.shape[0]
    assert s.shape == af.shape == bf.shape == ad.shape == bd.shape == (n, n)
    f_avg = (af + bf) / 2.
    p_d = projection(s, bd)
    p_s = projection(s, ad - bd)
    p_v = projection_complement(s, ad)
    f = (+ fock_block_sum(bf, [(p_d, p_s), (p_s, p_d)])
         + fock_block_sum(af, [(p_s, p_v), (p_v, p_s)])
         + fock_block_sum(f_avg, [(p_d, p_d), (p_s, p_s), (p_v, p_v),
                                  (p_d, p_v), (p_v, p_d)]))
    return f


def energy(h, g, ad, bd):
    aw, bw = mean_field(g, ad, bd)
    return np.sum((h + aw / 2) * ad + (h + bw / 2) * bd)


def dipole_moment(p, ad, bd):
    return np.tensordot(p, ad + bd, axes=[(1, 2), (0, 1)])


def uhf_orb_grad(s, af, bf, ad, bd):
    ap_o = projection(s, ad)
    ap_v = projection_complement(s, ad)
    bp_o = projection(s, bd)
    bp_v = projection_complement(s, bd)
    ar = fock_block_sum(af, [(ap_v, ap_o), (ap_o, ap_v)], weights=(1, -1))
    br = fock_block_sum(bf, [(bp_v, bp_o), (bp_o, bp_v)], weights=(1, -1))
    return np.array([ar, br])


def rohf_orb_grad(s, af, bf, ad, bd):
    p_d = projection(s, bd)
    p_s = projection(s, ad - bd)
    p_v = projection_complement(s, ad)
    f_avg = (af + bf) / 2.
    r = (+ fock_block_sum(f_avg, [(p_v, p_d), (p_d, p_v)], weights=(1, -1))
         + fock_block_sum(bf, [(p_s, p_d), (p_d, p_s)], weights=(1, -1))
         + fock_block_sum(af, [(p_v, p_s), (p_s, p_v)], weights=(1, -1)))
    return r


def mo_coefficients(s, f):
    _, c = spla.eigh(f, b=s)
    return c


def uhf_mo_coefficients(s, h, g, na, nb, guess_density=None, niter=100,
                        e_threshold=1e-10, d_threshold=1e-8, print_conv=False,
                        diis_start=3, ndiis_vecs=6, diis_extrapolator=None):
    nbf = s.shape[0]
    if guess_density is None:
        guess_density = np.zeros((2, nbf, nbf))

    assert nbf >= na >= nb
    assert s.shape == h.shape == (nbf, nbf)
    assert g.shape == (nbf, nbf, nbf, nbf)
    assert guess_density.shape == (2, nbf, nbf)

    ad, bd = guess_density

    e = e0 = de = r_norm = 0.
    iteration = 0
    converged = False
    f_series = []
    r_series = []
    for iteration in range(niter):
        # Update orbitals
        af, bf = fock(h, g, ad, bd)

        ac = mo_coefficients(s, af)
        bc = mo_coefficients(s, bf)
        ad = density(ac, slc=slice(na))
        bd = density(bc, slc=slice(nb))

        # Get energy change
        e = energy(h, g, ad, bd)
        de = np.fabs(e - e0)
        e0 = e

        # Get orbital gradient (MO basis)
        ar, br = uhf_orb_grad(s, af, bf, ad, bd)
        r_norm = spla.norm([ar, br])

        # Check convergence
        converged = (de < e_threshold and r_norm < d_threshold)
        if converged:
            break

        if callable(diis_extrapolator):
            if len(f_series) >= ndiis_vecs - 1:
                f_series.pop(0)
                r_series.pop(0)
            f_series.append((af, bf))
            r_series.append((ar, br))
            if len(f_series) >= diis_start:
                af, bf = diis_extrapolator(f_series, r_series)
                ac = mo_coefficients(s, af)
                bc = mo_coefficients(s, bf)
                ad = density(ac, slc=slice(na))
                bd = density(bc, slc=slice(nb))

    if not converged:
        warnings.warn("Did not converge! (dE: {:7.1e}, orb grad: {:7.1e})"
                      .format(de, r_norm))

    if print_conv:
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, orb grad: {:7.1e})"
              .format(e, iteration, de, r_norm))

    return np.array([ac, bc])


def rohf_mo_coefficients(s, h, g, na, nb, guess_density=None, niter=100,
                         e_threshold=1e-10, d_threshold=1e-8, print_conv=False,
                         diis_start=3, ndiis_vecs=6, diis_extrapolator=None):
    nbf = s.shape[0]
    if guess_density is None:
        guess_density = np.zeros((2, nbf, nbf))

    assert nbf >= na >= nb
    assert s.shape == h.shape == (nbf, nbf)
    assert g.shape == (nbf, nbf, nbf, nbf)
    assert guess_density.shape == (2, nbf, nbf)

    ad, bd = ad0, bd0 = guess_density

    e = e0 = de = r_norm = 0.
    iteration = 0
    converged = False
    f_series = []
    r_series = []
    for iteration in range(niter):
        # Update orbitals
        af, bf = fock(h, g, ad, bd)
        f = rohf_fock(s, af, bf, ad, bd)
        c = mo_coefficients(s, f)
        ad = density(c, slc=slice(na))
        bd = density(c, slc=slice(nb))

        # Get energy change
        e = energy(h, g, ad, bd)
        de = np.fabs(e - e0)
        e0 = e

        # Get the outer_square change
        d_norm = np.sqrt(spla.norm(ad - ad0) ** 2 + spla.norm(bd - bd0) ** 2)
        ad0 = ad
        bd0 = bd

        # Get orbital gradient (MO basis)
        r = rohf_orb_grad(s, af, bf, ad, bd)
        r_norm = spla.norm(r)

        # Check convergence
        converged = (de < e_threshold and d_norm < d_threshold)
        if converged:
            break

        if callable(diis_extrapolator):
            if len(f_series) >= ndiis_vecs - 1:
                f_series.pop(0)
                r_series.pop(0)
            f_series.append((f,))
            r_series.append((r,))
            if len(f_series) >= diis_start:
                f, = diis_extrapolator(f_series, r_series)
                c = mo_coefficients(s, f)
                ad = density(c, slc=slice(na))
                bd = density(c, slc=slice(nb))

    if not converged:
        warnings.warn("Did not converge! (dE: {:7.1e}, orb grad: {:7.1e})"
                      .format(de, d_norm))

    if print_conv:
        print("E={:20.15f} ({:-3d} iterations, dE: {:7.1e}, orb grad: {:7.1e})"
              .format(e, iteration, de, d_norm))

    return np.array([c, c])

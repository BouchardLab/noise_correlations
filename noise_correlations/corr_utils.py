import numpy as np
from numba import jit


@jit(nopython=True)
def gen_uniform_correlation(d, rng=None, return_betas=False):
    """Generate a correlation matrix from the uniform distribution.

    Based on:
    https://www.rdocumentation.org/packages/clusterGeneration/versions/1.3.7
    and
    Joe, H. (2006). Generating random correlation matrices based on partial correlations.
    Journal of Multivariate Analysis, 97(10), 2177-2189.

    Parameters
    ----------
    d : int
        Dimension for the matrix.
    rng : RandomState or None
        Random state.
    return_betas : bool
        For testing, if True, return the random betas used to generate the matrix.
    """
    if rng is None:
        rng = np.random
    if d == 1:
        return np.ones((1, 1))
    elif d == 2:
        rho = np.random.uniform(-1, 1)
        return np.array([[1, rho], [rho, 1]])
    else:
        if return_betas:
            betas = None
        mat = np.eye(d)
        alp = 1. + (d - 2.) / 2.
        for ii in range(d - 1):
            beta = rng.beta(alp, alp)
            if return_betas:
                betas.append(beta)
            val = 2. * beta - 1.
            mat[ii, ii + 1] = val
            mat[ii + 1, ii] = val
        for mm in range(2, d):
            alp = 1. + (d - 1 - mm) / 2.
            for jj in range(d - mm):
                sub = mat[jj:jj + mm + 1, jj:jj + mm + 1]
                b = sub.shape[0]
                r1 = sub[1:-1, 0]
                r3 = sub[1:-1, b - 1]
                R2 = sub[1:-1, 1:-1]
                Ri3 = np.linalg.solve(R2, r3)
                Ri1 = np.linalg.solve(R2, r1)
                beta = rng.beta(alp, alp)
                if return_betas:
                    betas.append(beta)
                rcond = 2. * beta - 1.
                term13 = r1 @ Ri3
                term11 = r1 @ Ri1
                term33 = r3 @ Ri3
                val = term13 + rcond * np.sqrt((1. - term11) * (1. - term33))
                mat[jj, jj + mm] = val
                mat[jj + mm, jj] = val
    if return_betas:
        betas = np.array(betas)
        return mat, betas
    return mat

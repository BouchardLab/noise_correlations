import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.special import factorial

from .discriminability import lfi_data, lda_data
from .null_models import random_rotation_data, shuffle_data
from . import null_models
from . import utils


def scatter_max_ps(Yp, n_boot, ps=None, faxes=None):
    if faxes is None:
        faxes = plt.subplots(1, 2, figsize=(10, 5))
    f, (ax0, ax1) = faxes
    if ps is None:
        n_neurons, n_angles, n_trials = Yp.shape
        n_pairs = (factorial(n_neurons, exact=True) //
                   (2 * factorial(n_neurons - 2, exact=True)))
        ps = np.full((2, 2, n_pairs, n_angles), np.nan)
        pair_idx = 0
        for ii in range(1, n_neurons):
            print(pair_idx, n_pairs)
            for jj in range(ii):
                for kk in range(n_angles):
                    x = Yp[[ii, jj]][:, kk].T
                    y = Yp[[ii, jj]][:, (kk + 1) % n_angles].T
                    val_s, values_s, ps_s = null_models.eval_null_data(x, y, shuffle_data, [lfi_data, lda_data], n_boot)
                    val_r, values_r, ps_r = null_models.eval_null_data(x, y, random_rotation_data, [lfi_data, lda_data], n_boot)
                    ps[0, :, pair_idx, kk] = ps_s
                    ps[1, :, pair_idx, kk] = ps_r
                pair_idx += 1
    plot_pvalue_comparison(ps[0, 0].ravel(), ps[1, 0].ravel(),
                           labels=['Shuffle p-value', 'Rotation p-value'], faxes=(f, ax0))
    plot_pvalue_comparison(ps[0, 1].ravel(), ps[1, 1].ravel(),
                           labels=['Shuffle p-value', 'Rotation p-value'], faxes=(f, ax1))
    ax0.set_title('LFI')
    ax1.set_title('LDA Accuracy')
    return ps, faxes


def scatter_blanche_ps(Yp, n_boot, ps=None, faxes=None):
    if faxes is None:
        faxes = plt.subplots(1, 2, figsize=(10, 5))
    f, (ax0, ax1) = faxes
    if ps is None:
        n_neurons, n_angles, n_trials = Yp.shape
        n_pairs = (factorial(n_neurons, exact=True) //
                   (2 * factorial(n_neurons - 2, exact=True)))
        ps = np.full((2, 2, n_pairs, n_angles), np.nan)
        pair_idx = 0
        for ii in range(1, n_neurons):
            print(pair_idx, n_pairs)
            for jj in range(ii):
                for kk in range(n_angles):
                    x = Yp[[ii, jj]][:, kk].T
                    y = Yp[[ii, jj]][:, (kk + 1) % n_angles].T
                    val_s, values_s, ps_s = null_models.eval_null_data(x, y, shuffle_data, [lfi_data, lda_data], n_boot)
                    val_r, values_r, ps_r = null_models.eval_null_data(x, y, random_rotation_data, [lfi_data, lda_data], n_boot)
                    ps[0, :, pair_idx, kk] = ps_s
                    ps[1, :, pair_idx, kk] = ps_r
                pair_idx += 1
    plot_pvalue_comparison(ps[0, 0].ravel(), ps[1, 0].ravel(),
                           labels=['Shuffle p-value', 'Rotation p-value'], faxes=(f, ax0))
    plot_pvalue_comparison(ps[0, 1].ravel(), ps[1, 1].ravel(),
                           labels=['Shuffle p-value', 'Rotation p-value'], faxes=(f, ax1))
    ax0.set_title('LFI')
    ax1.set_title('LDA Accuracy')
    return ps, faxes


def plot_pvalue_comparison(
    p0s, p1s, labels=None, fax=None, ax_lim=None, pval=0.05, cs=None,
    heatmap=False, show_inset=True, color_regions=True, insetfontsize=8
):
    """Plots the p-value comparison between two null models.

    Parameters
    ----------
    p0s, p1s : ndarray, (configurations,)
        The p-values across dimlet and stimulus-pair configurations. The former
        will be on the x-axis, while the latter is on the y-axis.
    labels : list of strings
        The labels for the x- and y-axes, respectively.
    fax : tuple of mpl.figure and mpl.axes, or None
        The figure and axes. If None, a new set will be created.
    ax_lim : float, default None
        The left (x-axis) and bottom (y-axis) limit for the axes. If None, the
        minimum p-value among all points will be used.
    pval : float, default 0.05
        The significance value to base plot regions on.
    cs : string, list of strings, or None
        Denotes which colors to use. If string, will use a pre-defined colormap.
        If list of strings, must contain 3 hex color codes, denoting the
        [top left region, bottom right region, bottom left region].
    heatmap : bool
        If True, uses a hexbin to show distribution of p-values. If False,
        a scatter plot is used.
    show_inset : bool
        If True, inset showing distribution of points in regions is depicted.
    color_regions : bool
        If True, colors specific regions of the p-value plot.
    insentfontsize : float
        The font size for the inset showing distribution of points.

    Returns
    -------
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects, with p-values plotted.
    """
    fig, ax = utils.check_fax(fax, n_rows=1, n_cols=1, figsize=(5, 5))

    # handle colors
    if cs is None:
        cs = [u'#9467bd', u'#8c564b', u'#17becf']
    elif cs == 'measure':
        cs = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
    elif not isinstance(cs, list):
        raise ValueError('Improper color key.')

    p0s[p0s == 0] = p0s[p0s > 0].min()
    p1s[p1s == 0] = p1s[p1s > 0].min()
    # plot the p-values according to hexbin or scatter plot
    if heatmap:
        ax.hexbin(p0s, p1s, cmap='gray_r', mincnt=1, xscale='log', yscale='log',
                  extent=[-4, 1, -4, 1], gridsize=40, bins='log')
    else:
        ax.scatter(p0s, p1s, marker='.', c='k')
    # choose axis bound if not provided
    if ax_lim is None:
        ax_lim = np.power(10., np.floor(np.log10(min(p0s.min(), p1s.min()))))
    # identity line
    ax.plot([ax_lim, 1], [ax_lim, 1], c='k')
    # significance at chosen p-value lines
    ax.axhline(pval, ax_lim, 1, c='k', ls='--')
    ax.axvline(pval, ax_lim, 1, c='k', ls='--')
    # stylize axes
    ax.set_xlim(ax_lim, 1)
    ax.set_ylim(ax_lim, 1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    # add colored rectangles denoting separate regions
    if color_regions:
        ax.add_artist(Rectangle((ax_lim, pval),
                                pval - min(pval, ax_lim),
                                1 - min(pval, ax_lim),
                                facecolor=cs[0], alpha=.3))
        ax.add_artist(Rectangle((pval, ax_lim),
                                1 - min(pval, ax_lim),
                                pval - min(pval, ax_lim),
                                facecolor=cs[1], alpha=.3))
        ax.add_artist(Rectangle((ax_lim, ax_lim),
                                pval - min(pval, ax_lim),
                                pval - min(pval, ax_lim),
                                facecolor=cs[2], alpha=.3))

    # show inset with bar plot showing proportions of points in regions
    if show_inset:
        pos = ax.get_position()
        # width and height of inset
        w = pos.x1 - pos.x0
        h = pos.y1 - pos.y0
        edge = .175
        size = .35
        # add inset axes
        inset = fig.add_axes([pos.x0 + edge * w, pos.y0 + edge * h / 2.,
                              size * w, size * h])
        # calculate fraction in each region
        total = p0s.size
        p0_not_1 = (np.logical_and(p0s < .05, p1s >= .05)).sum() / total
        p1_not_0 = (np.logical_and(p1s < .05, p0s >= .05)).sum() / total
        p0_and_1 = (np.logical_and(p1s < .05, p0s < .05)).sum() / total
        max_proportion = 1.05 * max(p0_not_1, p1_not_0, p0_and_1)
        # bar plot depicting these proportions
        inset.bar([0, 1, 2], [p0_not_1, p1_not_0, p0_and_1], color=cs, alpha=.3)
        # stylize inset axes
        inset.set_yticks([])
        inset.set_xticks([])
        inset.set_xticklabels(['', ''])
        inset.set_xlabel('Total: {}'.format(total), fontsize=insetfontsize)
        inset.set_ylabel('Fraction', fontsize=insetfontsize)
        inset.set_ylim([0, 1.10 * max_proportion])
        # insert text indiciating proportions above each bar
        inset.text(0, 1.075 * max_proportion, s='%.2f' % p0_not_1,
                   va='top', ha='center',
                   fontsize=insetfontsize)
        inset.text(1, 1.075 * max_proportion, s='%.2f' % p1_not_0,
                   va='top', ha='center',
                   fontsize=insetfontsize)
        inset.text(2, 1.075 * max_proportion, s='%.2f' % p0_and_1,
                   va='top', ha='center',
                   fontsize=insetfontsize)

    return fig, ax


def plot_tuning_curves(
    X, stimuli, n_cols=5, fax=None, include_points=False, use_title=False,
    sort=None
):
    """Plots a set of tuning curves given a neural design matrix.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    n_cols : int
        The number of columns in the subplot grid. Ignored if fax is not None.
    fax : tuple of mpl.figure and mpl.axes, or None
        The figure and axes. If None, a new set will be created.
    include_points : bool
        If True, the individual samples are included.

    Returns
    -------
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects, with tuning curves plotted.
    """
    n_samples, n_units, n_stimuli, unique_stimuli = utils.X_stimuli(X, stimuli)
    # calculate number of rows and create figure axes
    n_rows = np.ceil(n_units / n_cols)
    fig, axes = utils.check_fax(fax, n_rows=n_rows, n_cols=n_cols,
                                figsize=(1.5 * n_cols, 1.5 * n_rows))

    # calculate tuning curves
    tuning_curves = utils.get_tuning_curve(X, stimuli, aggregator=np.mean)

    # sort units by peak response
    if sort == 'peak':
        peak_responses = utils.get_peak_response(X, stimuli)
        sorted_units = np.argsort(peak_responses)
    elif sort == 'modulation':
        modulations = utils.get_tuning_modulation(X, stimuli)
        sorted_units = np.argsort(modulations)
    elif sort is None:
        sorted_units = np.arange(n_units)
    else:
        raise ValueError('Improper value for sort.')

    # plot each tuning curve
    for ax_idx, ax in enumerate(axes.ravel()):
        # turn off subplot if we've gone past the number of units
        if ax_idx >= n_units:
            ax.axis('off')
        else:
            unit_idx = sorted_units[ax_idx]
            # plot individual samples if necessary
            if include_points:
                # iterate over each stimulus
                for idx, stimulus in enumerate(unique_stimuli):
                    # plot all samples for a given stimulus
                    sample_idx = stimuli == stimulus
                    ax.scatter(
                        stimulus * np.ones(sample_idx.sum()),
                        X[sample_idx][:, unit_idx],
                        color='gray',
                        s=20,
                        alpha=0.5)
            tuning_curve = tuning_curves[unit_idx]
            # plot tuning curve
            ax.plot(unique_stimuli, tuning_curve,
                    color='black', linewidth=3, marker='o')
            if use_title:
                title = 'R: %0.2f \n M: %0.2f' % (
                    np.max(tuning_curve), np.max(tuning_curve) - np.min(tuning_curve)
                )
                ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([unique_stimuli[0], unique_stimuli[-1]])
    return fig, axes


def plot_ellipses(mu0, cov0, mu1, cov1, ld_cov=None, faxes=None, alpha=.5):
    """Plot ellipses corresponding to bivariate normal distributions
    with means mu0, mu1 and covariances cov0, cov1. Can also include
    an ellipse for the linear discriminability covariance.

    Parameters
    ----------
    mu0 : ndarray (2,)
    cov0 : ndarray (2, 2)
    mu1 : ndarray (2,)
    cov1: ndarray (2, 2)
    ld_cov : ndarray (2, 2)
    """
    if mu0.size != 2:
        raise ValueError
    if faxes is None:
        faxes = plt.subplots(1, figsize=(5, 5))
    f, ax = faxes
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    for mu, cov, c in [(mu0, cov0, c0), (mu1, cov1, c1)]:
        e, v = np.linalg.eigh(cov)
        e = np.sqrt(e)
        ell = Ellipse(mu, e[1], e[0],
                      180. * np.arctan2(v[1, -1], v[0, -1]) / np.pi,
                      facecolor=c, alpha=alpha)
        ax.plot(mu[0], mu[1], 'o', c=c)
        ax.add_artist(ell)
    if ld_cov is not None:
        e, v = np.linalg.eigh(ld_cov)
        e = np.sqrt(e)
        ell = Ellipse(.5 * (mu0 + mu1), e[1], e[0],
                      180. * np.arctan2(v[0, -1], v[1, -1]) / np.pi,
                      facecolor='None', alpha=alpha, edgecolor='k')
        ax.add_artist(ell)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    return


def histo_samples(orig0, orig1, trans, measure, nsamples, faxes=None):
    """
    Plot a histogram of nsamples values of measure evaluated on orig0 and orig1
    after applying trans. Original value plotted as vertical line.

    Parameters:
    -----------
    orig0       (examples, dim) data
    orig1       (examples, dim) data
    trans       (callable) data transformation
    measure     (callable) takes 2 data arguments, returns comparison measure
    nsamples    (int) number of times to apply trans and evaluate measure

    Returns:
    measure(orig0, orig1)
    fraction of samples with measure less than original
    matplotlib axis object
    """
    values = np.zeros(nsamples)
    for ii in range(nsamples):
        new0 = trans(orig0)
        new1 = trans(orig1)
        values[ii] = measure(new0, new1)
    if faxes is None:
        faxes = plt.subplots(1, 1)
    fig, ax = faxes
    vals, bins, patches = ax.hist(values, bins=50, density=True)
    orig_val = measure(orig0, orig1)
    ax.vlines(orig_val, 0, np.max(vals))
    frac_less = np.count_nonzero(orig_val > values) / nsamples
    return orig_val, frac_less, ax


def rot_plot(mu0, cov0, mu1, cov1, measure, nsamples=10000):
    """Plots comparison to random-rotation null model."""
    def unpack_rot(args):
        return random_rotation(*args)

    def unpack_measure(arg0, arg1):
        return measure(*arg0, *arg1)
    val, frac, ax = histo_samples((mu0, cov0), (mu1, cov1),
                                  unpack_rot, unpack_measure, nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax


def ellipses_and_measure_normal_data(x0, x1):
    mu0 = x0.mean(0)
    cov0 = np.cov(x0)
    mu1 = x1.mean(0)
    cov1 = np.cov(x1)
    discriminability.plot_ellipses(mu0, cov0, mu1, cov1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys(mu0, cov0,
                                                           mu1, cov1))


def scatter_and_measures(x0, x1):
    fig, ax = plt.subplots(1)
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    ax.plot(x0[:, 0], x0[:, 1], '.', color=c0)
    ax.plot(x1[:, 0], x1[:, 1], '.', color=c1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys_data(x0, x1))
    print('Linear discriminability',
          discriminability.linear_discriminability_data(x0, x1))
    print('Quadratic disciminability', discriminability.qda_data(x0, x1))


def shuffle_plot_data(x0, x1, measure, ntrials=1000):
    val, frac = histo_samples(x0, x1, shuffle_data, measure, ntrials)
    print('Value:', val)
    print('Fraction of shuffles giving smaller values: ', frac)


def rot_plot_data(x0, x1, measure, nsamples=10000):
    val, frac, ax = histo_samples(x0, x1, random_rotation_data, measure,
                                  nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax


def median_ps(ps, dim, fax=None, label=None, dx=0., c='C0'):
    if fax is None:
        fax = plt.subplots(1)
    f, ax = fax
    ps = ps.copy()
    ps[ps==0] = 1. / 1e4
    ps = np.log10(ps)
    med = np.median(ps)
    yerr = abs(np.percentile(ps, [16, 84])[:, np.newaxis] - med)
    ax.errorbar(dim - dx, med, yerr=yerr, c=c, fmt='.', label=label)

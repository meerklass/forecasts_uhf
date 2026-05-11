import numpy as np
import healpy as hp
import os

from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d
from meer21cm import MockSimulation
from meer21cm.util import redshift_to_freq

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])
    
    return spline

######################
# BINNING 3D SPECTRA #
######################

def bin_power_cy(
    power_3d,
    k_perp,
    k_para,
    kperpbins,
    kparabins,
    kweights=None,
):
    '''
    Bins a 3D power spectrum in k_perp k_par bins
    '''
    pcy_arr = bin_3d_to_cy(
        power_3d, k_perp, kperpbins, vectorize=True,
        weights=kweights,
    )
    pcy_arr = bin_3d_to_cy(
        np.nan_to_num(pcy_arr), np.abs(k_para), kparabins, vectorize=True,
        weights=(1-np.isnan(pcy_arr))[0].astype('float'),
    )
    return pcy_arr

def bin_power_1d(
    power_3d,
    k_mode,
    k1dbins,
    kweights,
    num_split=None,
):
    '''
    Bins a 3D power spectrum in k bins
    '''
    if num_split is None:
        p1d, keff, nmodes = bin_3d_to_1d(
            power_3d, k_mode, k1dbins, vectorize=True,weights=kweights,
        )
    else:
        p1d = []
        power_3d_arr = np.array_split(power_3d,num_split)
        for i in range(num_split):
            pdata1darr_i, keff, nmodes = bin_3d_to_1d(
                power_3d_arr[i], k_mode, k1dbins, vectorize=True,weights=kweights,
            )
            p1d.append(pdata1darr_i)
        p1d = np.concatenate(p1d)
    return p1d, keff, nmodes

####################
# Ploting routines #
####################


def plot_cy_power(xbins, ybins, pdatacy, pmodcy, vmin_ratio, vmax_ratio):
    arr = np.array(
        [
            np.log10(pdatacy.T),
            np.log10(pmodcy.T),
        ]
    )
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        xbins,
        ybins,
        np.log10(pdatacy.T),
        vmin=vmin,
        vmax=vmax,
    )
    im = axes[1].pcolormesh(
        xbins,
        ybins,
        np.log10(pmodcy.T),
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=axes[:-1], location="top", fraction=0.046, pad=0.04)
    im = axes[2].pcolormesh(
        xbins,
        ybins,
        (pdatacy.T) / (pmodcy.T),
        vmin=vmin_ratio,
        vmax=vmax_ratio,
        cmap="bwr",
    )
    plt.colorbar(im, ax=axes[2], location="top", fraction=0.046, pad=0.04)
    return fig


def plot_1d_power(
    keff,
    pdatad,
    pmodd,
    ratio_min,
    ratio_max,
):
    keff = np.array(keff)
    pdatad = np.array(pdatad)
    pmodd = np.array(pmodd)
    sel = keff == keff
    keff = keff[sel]
    pdatad = pdatad[:, sel]
    pmodd = pmodd[sel]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        height_ratios=[2, 1],
    )
    axes[0].errorbar(
        keff,
        pdatad.mean(axis=0) * keff,
        yerr=pdatad.std(axis=0) * keff,
        label="mock",
    )
    axes[0].plot(keff, pmodd * keff, label="model", ls="--")
    axes[0].set_ylim((pmodd * keff).min() * 0.7, (pmodd * keff).max() * 1.2)
    axes[0].legend()
    axes[1].errorbar(
        keff,
        (pdatad.mean(axis=0)) / (pmodd) - 1,
        yerr=(pdatad.std(axis=0)) / (pmodd),
    )
    axes[1].axhline(0, color="black", ls="--")
    axes[1].fill_between(
        np.linspace(keff.min() - 0.005, keff.max() + 0.005, 100),
        -0.05,
        0.05,
        color="black",
        alpha=0.2,
    )
    axes[1].set_xlim(keff.min() - 0.005, keff.max() + 0.005)
    axes[1].set_ylim(ratio_min, ratio_max)
    axes[1].legend()
    return fig

def generate_healpix_mask(dic):
    # extract the basic metadata
    nu_min = redshift_to_freq(dic['z_max'])
    nu_max = redshift_to_freq(dic['z_min'])
    
    ra_range = [dic['ra_center'] - dic['ra_obs_width']/2, dic['ra_center'] + dic['ra_obs_width']/2]
    dec_range = [dic['dec_center'] - dic['dec_obs_width']/2, dic['dec_center'] + dic['dec_obs_width']/2]
    
    sim_upres_radial = 1
    sim_upres_transvers = 1/2
    
    # Load galaxies #
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        dic['dndz_filename'],
    )
    dndz_data = np.load(path)
    
    z_bin = dndz_data["z_bin"]
    z_count = dndz_data["z_count"]
    z_cen = (z_bin[:-1] + z_bin[1:]) / 2
    dV_arr = dic['Cosmo'].differential_comoving_volume(z_cen)
    
    zgal_func = interp1d(
            z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
        )
    
    mock = MockSimulation(
        nu_min=nu_min,
        nu_max=nu_max,
        pickle_file=dic['pickle_file'],
        downres_factor_radial=sim_upres_radial,
        downres_factor_transverse=sim_upres_transvers,
        batch_number=dic['batch_number'],
        discrete_source_dndz=zgal_func,
        tracer_bias_2=1.0,
        tracer_bias_1=1.0,
        sigma_v_1=100,
        sigma_v_2=100,
        mean_amp_1="average_hi_temp",
    )
    mock.read_from_pickle()
    #mock.nu = mock.nu[::downres_factor_freq] # no need for extremely fine frequency resolution
    #mock.W_HI = mock.W_HI[:,:,::downres_factor_freq]
    #mock.w_HI = mock.w_HI[:,:,::downres_factor_freq]
    mock.data = np.zeros((mock.num_pix_x,mock.num_pix_y,mock.nu.size))
    mock.trim_map_to_range()
    mock_hp = MockSimulation(
        hp_nside=dic['hp_nside'], 
        nu = mock.nu,
        ra_range = ra_range, 
        dec_range = dec_range, 
    )
    pixel_id_wcs = hp.ang2pix(mock_hp.hp_nside,mock.ra_map,mock.dec_map,lonlat=True)
    hit_counts_hp = np.zeros((hp.nside2npix(mock_hp.hp_nside),len(mock.nu)))
    for i in range(len(mock.nu)):
        np.add.at(hit_counts_hp[:,i],pixel_id_wcs.ravel(),mock.w_HI[:,:,i].ravel())
        # smooth it
        map_temp = hp.ud_grade(hit_counts_hp[:,i],mock_hp.hp_nside//2)
        hit_counts_hp[:,i] = hp.ud_grade(map_temp,mock_hp.hp_nside)
    hit_counts_hp = hit_counts_hp[mock_hp.pixel_id]
    return mock.nu, hit_counts_hp

def generate_mock_healpix(nu=None,hit_counts_hp=None, dic = None):
    if hit_counts_hp is None:
        nu, hit_counts_hp = generate_healpix_mask()
    ra_range = [dic['ra_center'] - dic['ra_obs_width']/2, dic['ra_center'] + dic['ra_obs_width']/2]
    dec_range = [dic['dec_center'] - dic['dec_obs_width']/2, dic['dec_center'] + dic['dec_obs_width']/2]
    mock = MockSimulation(
        hp_nside=dic['hp_nside'],
        nu = nu,
        ra_range = ra_range,
        dec_range = dec_range,
    )
    mock.W_HI = hit_counts_hp>0
    mock.w_HI = hit_counts_hp
    return mock
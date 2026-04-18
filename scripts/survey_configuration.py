import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from meer21cm.telescope import dish_beam_sigma
from meer21cm.util import create_wcs, freq_to_redshift, redshift_to_freq
from scipy.interpolate import CubicSpline, interp1d

from utils import add_boundary_knots

####################
# INPUT PARAMETERS #
####################

z_min, z_max = 0.6, 0.66                    # Minimum and maximum redshift to observe 
pix_resol = 0.5                             # Pixel resolution (in degrees)
nu_resol = 132812.5                         # Frequency resolution --- channel width (in Hz)
#
ra_center = 150                             # Center RA position (in degrees)
ra_sky_width = 60                           # Width in RA for the sky WCS object -> NOT SURVEY! (in degrees)
dec_center = -2.5                           # Center DEC position (in degrees)
dec_sky_width = 20                          # Width in DEC for the sky WCS object -> NOT SURVEY! (in degrees)
#
ra_obs_width = 50                           # Width in RA for the observed patch (in degrees)
dec_obs_width = 15                          # Width in DEC for the observed patch (in degrees)
#
dish_size = 13.5                            # Antenna diameter (in meters)
ndish = 64                                  # Number of dishes observing
t_obs = 20                                  # Total observing time per dish (in hours)
n_feeds = 2                                 # Number of detectors per dish
#
window_name = "blackmanharris"              # Name for tapering window
#
dndz_filename = "LRGELG_dndz.npz"           # Name for the galaxy dndz file. It assumes the file is on the same folder as this file
ngal = 771875 / 4 / 1e9                     # galaxy total number density in Mpc^-3 (this number is for DESI DR1 LRG2 bin)
# 
Cosmo = Planck18                            # Cosmology astropy object -> model and specific parameter values
#
kmin, kmax = 0.003, 0.2                     # Minimum and maximum k values for the 1D power spectrum 
nk = 9                                      # Number of k bins for the 1D power spectrum
kmax_par, kmax_perp = 0.5, 0.05             # Maximum k values for the parallel and perpendicular directions
nk_par, nk_perp = 27, 11                    # Number of k bins for the parallel and perpendicular k values
#
sigma_v_1, sigma_v_2 = 100,100              # typical peculiar velocity for tracers 1/2 for the RSD FoGs (in km/s)
tracer_bias_1, tracer_bias_2 = 1.5, 2.      # bias for the tracers 1/2
Omega_HI = 5e-4                             # Omega_HI at z = 0 abundance
mean_amp_1 = "average_hi_temp"              # Which amplitude to use for the HI field
####
# If you want a different matter power spectrum (example commented below for a power law for a scale-free simulation)
'''
kk = np.geomspace(1e-4,1e3,100)
pk = kk**-2
ipk = interp1d(kk,pk,bounds_error=False,fill_value='extrapolate')
'''

# ==============================================================
### From here one, you should have no need to modify anything
# ==============================================================

#################################
# Frequency & redshift channels # 
#################################

nu_min = redshift_to_freq(z_max)
nu_max = redshift_to_freq(z_min)
num_ch = int((nu_max - nu_min) / nu_resol)
nu_arr = np.linspace(nu_min, nu_min + (num_ch - 1) * nu_resol, num_ch)
z_ch = freq_to_redshift(nu_arr)

#####################
# CREATE WCS OBJECT # 
#####################

num_pix_x = int(ra_sky_width / pix_resol)
num_pix_y = int(dec_sky_width / pix_resol)

wcs = create_wcs(
    ra_cr=ra_center,
    dec_cr=dec_center,
    ngrid=[num_pix_x, num_pix_y],
    resol=pix_resol,
)

ra_range = [ra_center - ra_obs_width/2, ra_center + ra_obs_width/2]
dec_range = [dec_center - dec_obs_width/2, dec_center + dec_obs_width/2]

######################
# Angular resolution #
######################

_sigma_beam_ch = dish_beam_sigma(dish_size, nu_arr)
_comov_dist = Cosmo.comoving_distance(z_ch).value
sigma_beam_new = 1 / _comov_dist * _sigma_beam_ch
sigma_beam_new *= _sigma_beam_ch.mean() / sigma_beam_new.mean()

##########
# k bins #
##########

k1d_bins = np.linspace(kmin, kmax, nk+1)[1:]
kperp_bins = np.linspace(0, kmax_perp, nk_perp+2)[2:]
kpar_bins = np.linspace(0, kmax_par, nk_par)[1:]

##################
# Detector noise #
##################

# MeerKLASS fit for system temperature
NU_MHZ = np.array([565.2928416485901, 578.3080260303688, 585.6832971800434, 591.7570498915401, 606.5075921908893, 616.9197396963124, 626.4642082429501, 631.236442516269, 643.3839479392625, 646.4208242950108, 654.2299349240781, 665.5097613882863, 677.6572668112798, 690.2386117136659, 704.5553145336225, 720.1735357917571, 738.82863340564, 751.8438177874186, 755.3145336225597, 768.763557483731, 791.7570498915402, 802.1691973969631, 820.824295010846, 837.7440347071583, 847.2885032537961, 859.002169197397, 868.5466377440348, 873.7527114967462, 884.1648590021691, 892.407809110629, 911.062906724512, 923.644251626898, 933.1887201735358, 960.9544468546637, 983.5140997830803, 996.9631236442517, 1011.7136659436009, 1030.3687635574838, 1047.288503253796, 1055.0976138828632, 1060.303687635575])

TSYS_OVER_ETA_K = np.array([36.75302245250432, 35.673575129533674, 34.98272884283247, 33.773747841105354, 33.1692573402418, 32.65112262521589, 32.089810017271155, 31.528497409326427, 31.355785837651123, 30.362694300518136, 29.32642487046632, 30.40587219343696, 29.585492227979273, 29.02417962003454, 27.858376511226254, 27.5993091537133, 27.08117443868739, 25.094991364421418, 26.260794473229705, 25.9153713298791, 25.310880829015545, 23.97236614853195, 23.97236614853195, 22.979274611398964, 23.238341968911918, 22.461139896373055, 21.8566493955095, 21.33851468048359, 19.8272884283247, 22.202072538860104, 21.8566493955095, 21.986183074265977, 21.07944732297064, 20.77720207253886, 20.129533678756477, 19.654576856649395, 19.870466321243523, 20.08635578583765, 21.511226252158895, 23.324697754749568, 28.808290155440414])

tsys_inter = CubicSpline(NU_MHZ, TSYS_OVER_ETA_K, bc_type="natural")
tsys_inter = add_boundary_knots(tsys_inter)

def sigma_N(num_pix):
    nu = nu_arr * u.Hz
    dnu = nu_resol * u.Hz

    tsys_over_eta = tsys_inter(nu.to(u.MHz).value) * u.K

    t_tot = 20 * u.hr
    n_dish = 64
    n_feeds = 2
    t_pixel = n_dish * t_tot / num_pix

    return tsys_over_eta / np.sqrt(n_feeds * (dnu * t_pixel).to(1).value)

def sigma_N_of_freq(nu,num_pix):
    nu = nu * u.Hz
    dnu = nu_resol * u.Hz

    tsys_over_eta = tsys_inter(nu.to(u.MHz).value) * u.K

    t_tot = 20 * u.hr
    n_dish = 64
    n_feeds = 2
    t_pixel = n_dish * t_tot / num_pix

    return tsys_over_eta / np.sqrt(n_feeds * (dnu * t_pixel).to(1).value)

#################
# Load galaxies #
#################

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    dndz_filename,
)
dndz_data = np.load(path)

z_bin = dndz_data["z_bin"]
z_count = dndz_data["z_count"]
z_cen = (z_bin[:-1] + z_bin[1:]) / 2
dV_arr = Planck18.differential_comoving_volume(z_cen)

zgal_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )
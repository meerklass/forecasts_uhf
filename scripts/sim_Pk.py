import sys
import os 

sys.path.append(os.path.abspath("")+'/../scripts/')

import numpy as np
import scipy.signal.windows as windows

from meer21cm.grid import shot_noise_correction_from_gridding
from meer21cm.power import get_shot_noise_galaxy
from set_inputs import *


def run_realization_Pks(input_dict, default_dict):
    '''
    Return the power spectra of the lognormal realizations 
    for a given input stored in survey_configuration.py

    Input:
        - input_dict: whether you want to change some of the inputs
        - default_dict: defined in set_inputs
        - seed: RNG seed
    Returns:
        mock.kmode, Pk_HI_3D, Pk_HIxgal_3D, Pk_gal_3D
    '''
    
    # If you want a different matter power spectrum (commented here, uncomment also in survey_configuration)
    '''
    kk = np.geomspace(1e-4,1e3,100)
    pk = kk**-2
    ipk = interp1d(kk,pk,bounds_error=False,fill_value='extrapolate')
    input_dict['ipk']
    '''

    mock, dic = set_meer21cm(input_dict = input_dict, default_dict = default_dict)

    # Get the tupper function
    mock.taper_func = getattr(windows, dic['window_name'])
    
    # Get total number of galaxies
    num_gal = int(mock.survey_volume * dic['ngal'])
    mock.num_discrete_source = num_gal

    #mock.W_HI = np.ones_like(mock.W_HI)
    #mock.w_HI = np.ones_like(mock.w_HI)
    
    # Get the enclosing box with better resolution
    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()

    #Make the 3D HI realization, go to sky to apply beam and trim, and come back to 3D

    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed+50) # this 50 means nothing, just want a different seed
    num_pix =np.sum(mock.W_HI[:,:,0]) #total number of pixels that we have in the WCS object
    noise_realisation = (generator.normal(scale = sigma_N(dic, num_pix)[None, None, :].to(u.K).value, size=(mock.num_pix_x, mock.num_pix_y, len(mock.nu))))

    mock.data = (
        mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
        + noise_realisation
    )
    
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.trim_map_to_range()

    # From the same underlying realization, obtain the galaxy distribution
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_gal_to_range()
    
    # restore window
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    
    # compute field from data and weights
    mock.grid_scheme = "cic"
    himap_rg, _, _ = mock.grid_data_to_field()
    
    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    
    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box 
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float')
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    # Compute the power spectra

    # Estimate galaxy shot noise
    shot_noise = (get_shot_noise_galaxy(galmap_rg, mock.box_len, mock.weights_grid_2, mock.weights_field_2,)
                * shot_noise_correction_from_gridding(mock.box_ndim, mock.grid_scheme))

    Pk_HI_3D = mock.auto_power_3d_1
    Pk_HIxgal_3D = mock.cross_power_3d
    Pk_gal_3D = mock.auto_power_3d_2 - shot_noise #We're removing shot_noise!!

    return mock.kmode, Pk_HI_3D, Pk_HIxgal_3D, Pk_gal_3D
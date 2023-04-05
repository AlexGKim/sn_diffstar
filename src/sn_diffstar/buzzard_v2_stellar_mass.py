from astropy.cosmology import FlatLambdaCDM
import fitsio
import numpy as np
import sys

def get_derived_quantities(template_path, coeff, z, cosmo):
    """
    Given a path to a kcorrect template file, as well as
    kcorrect coefficients for a set of galaxies and return derived
    quantities
    inputs
    ------
    template_path -- str
        path to kcorrect templates
    coeff -- (N x N_templates) array
        array of kcorrect coefficients
    z -- (N,) array
        redshifts of galaxies
    returns
    -------
    sfr_300 -- (N,) array
        Stellar mass formed in the last 300 Myr
    sfr_1000 -- (N,) array
        Stellar mass formed in the last 1000 Myr        
    smass -- (N,) array
        Array of stellar masses for galaxies
    """

    # read in relevant template info
    sfh_300 = fitsio.read(template_path, 19)
    sfh_1000 = fitsio.read(template_path, 20)
    sfh_met = fitsio.read(template_path, 13)
    mass_tot = fitsio.read(template_path, 17)

    # get angular diameter distances

    da = cosmo.angular_diameter_distance(z).value

    smass = np.dot(mass_tot, coeff.T) * (da * 1e6 / 10) ** 2
    sfr_300 = np.dot(coeff, sfh_300)
    sfr_1000 = np.dot(coeff, sfh_1000)

    return sfr_300, sfr_1000, smass

if __name__=='__main__':
    kfile = sys.argv[1] #name of file containing
    filename = sys.argv[2] #name of galaxy catalog file
    tfile = sys.argv[3] #name of galaxy catalog file
    
    cosmo = FlatLambdaCDM(100, 0.286)
    training_galaxies = fitsio.read(tfile)
    galaxies  = fitsio.FITS(filename)[-1].read(columns=['SEDID','Z','MAG_R'])
    coeffs = training_galaxies['COEFFS'][galaxies['SEDID']]
    mag_r_train = training_galaxies['ABSMAG'][galaxies['SEDID'],2] 
    
    coeffs *= 10 ** ((galaxies['MAG_R'] - mag_r_train.reshape(-1, 1)) / -2.5)
    
    sfr, met, smass = get_derived_quantities(kfile,coeffs,galaxies['Z'], cosmo)
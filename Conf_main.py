import numpy as np
import Confocal_Main_Func as confmain
import Confocal_Processing as proc
import Data_Check as dc
import matplotlib.pyplot as plt
import microscPSF as msPSF
from fractions import Fraction
from scipy import signal

# This program takes a 3D sample and simulates a stage scanning microscope. Hence the sample is 'moved'
# relative to the detector.
# The final output is a 3D image whereby the sample has passed through: a lens as denoted by a convolution with a
# radial PSF, and a pinhole simulated by a layer wise convolution with a gaussian spacial filter.
# Array sizes need to be odd to work with the convolution as a centre point needs to be determinable.

### INPUTS ###
#  Laser, PSF and Sample #
xy_size = 100           # xy size for both laser and sample.
pixel_size = 0.02      # Ground truth pixel size in microns
stack_size = 100         # Z depth for the PSF
laser_power = 1000000000000       # Laser power per second in should be microwatts where 1 count = 1 microwatt. average in live cell is 15000 microwatt
exposure_time = 1       # seconds of exposure
# PSF
excitation_wavelength = 0.480    # Wavelength in microns
emission_wavelength = 0.540      # Wavelength in microns
NA = 1.4                # Numerical aperture
### Dont touch the below line ###
msPSF.m_params["NA"] = NA   # alters the value of the microscope parameters in microscPSF. Has a default value of 1.4
# PINHOLE #
pinhole_radius = 100        # Radius of the pinhole in pixels.
offset = 0                # Offsets the pinhole. Meant to help increase resolution but needs to be implemented in fourier reweighting..
# CAMERA
camera_pixel_size = 6   # Camera pixel size in microns. usual sizes = 6 microns or 11 microns
magnification = 100     # Lens magnification
msPSF.m_params["M"] = magnification   # alters the value of the microscope parameters in microscPSF. Has a default value of 100
QE = 0.7                # Quantum Efficiency
gain = 2                # Camera gain. Usually 2 per incidence photon
count = 100             # Camera artificial count increase.
# NOISE
include_read_noise = "Y" # Y/N. Include read noise
read_mean = 2           # Read noise mean level
read_std = 2             # Read noise standard deviation level
include_shot_noise = "Y" # Y/N. Include shot noise
fix_seed = "Y"        # Y/N to fix the Shot noise seed.
include_fixed_pattern_noise = "Y" # Y/N. Include fixed pattern noise
fixed_pattern_deviation = 0.001  # Fixed pattern standard deviation. usually affects 0.1% of pixels.
# MODE #
mode = "Widefield"       # Mode refers to whether we are doing "Widefield NEED TO ADD", Confocal or ISM imaging.
# SAVE
Preview = "Y"
SAVE = "N"              # Save parameter, input Y to save, other parameters will not save.
filename = "Conf_pinhole100_200x200"   # filename without file format as a string, saves as tiff


### DATA CHECKS ###
xy_size = int(dc.simple_datacheck_lesser_greater(xy_size, 'xy_size', 1, np.inf))
pixel_size = dc.simple_datacheck_lesser_greater(pixel_size, "pixel_size (in microns)", 0.001, 1)
stack_size = int(dc.simple_datacheck_lesser_greater(stack_size, "stack_size", 1, np.inf))
laser_power = dc.simple_datacheck_lesser_greater(laser_power, "laser_poeer", 1, np.inf)
exposure_time = dc.simple_datacheck_lesser_greater(exposure_time, "exposure_time", 0, np.inf)
# PSF
emission_wavelength = dc.simple_datacheck_lesser_greater(emission_wavelength, "wavelength (in microns)", 0, 5)
excitation_wavelength = dc.simple_datacheck_lesser_greater(excitation_wavelength, "wavelength (in microns)", 0, 5)
NA = dc.simple_datacheck_lesser_greater(NA, "NA", 0.2, 2)
msPSF.m_params["NA"] = NA   # alters the value of the microscope parameters in microscPSF. Has a default value of 1.4
# PINHOLE #
pinhole_radius = dc.simple_datacheck_lesser_greater(pinhole_radius, "pinhole_radius", 0, xy_size)
offset = int(dc.simple_datacheck_lesser_greater(offset, "offset", 0, 10))
# CAMERA
camera_pixel_size = dc.simple_datacheck_lesser_greater(camera_pixel_size, "camera_pixel_size (in microns)", 1, 30)
magnification = dc.simple_datacheck_lesser_greater(magnification, "magnification", 1, 250)
QE = dc.simple_datacheck_lesser_greater(QE, "QE", 0, 1)
gain = dc.simple_datacheck_lesser_greater(gain, "gain", 1, 500)
count = dc.simple_datacheck_lesser_greater(count, "count", 0, 1000)
# NOISE
yes_no = ["Y", "N"]
include_read_noise = dc.simple_datacheck_string(include_read_noise, "include_read_noise", yes_no)
read_mean = dc.simple_datacheck_lesser_greater(read_mean, "read_mean", 0, 200)
read_std = dc.simple_datacheck_lesser_greater(read_std, "read_std", 0, 200)
include_shot_noise = dc.simple_datacheck_string(include_shot_noise, "include_shot_noise", yes_no)
fix_seed = dc.simple_datacheck_string(fix_seed, "fix_seed", yes_no)
include_fixed_pattern_noise = dc.simple_datacheck_string(include_fixed_pattern_noise, "include_fixed_pattern_noise", yes_no)
fixed_pattern_deviation = dc.simple_datacheck_lesser_greater(fixed_pattern_deviation, "fixed_pattern_deviation", 0, 1)
# MODE #
mode_options = ["Widefield", "Confocal", "ISM"]
mode = dc.simple_datacheck_string(mode, "mode", mode_options)
# SAVE #
yes_no = ["Y", "N"]
Preview = dc.simple_datacheck_string(Preview, "Preview", yes_no)
SAVE = dc.simple_datacheck_string(SAVE, "SAVE", yes_no)

print("DATA GOOD TO GO!")


###### MAIN PROGRAM ######
# Order of the program: produce sample and PSF, multiply, convolute and sum them in z for each xy point, bin the images
#                       as though they have been magnified to the cameras pixels, then process the data through
#                       confocal or ISM methods to produce the final image.

### PSF Generation ###
# Made a 3D PSF
# Each pixel = x nanometres.
# Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
laserPSF = confmain.radial_PSF(xy_size, pixel_size, stack_size, excitation_wavelength)
laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

laserPSF = (laserPSF / laserPSF.sum()) * (laser_power * exposure_time)  # Equating to 1. (to do: 1 count =  1 microwatt,
                                                                                        # hence conversion to photons.)


### SAMPLE PARAMETERS ###
if mode == "Widefield":
    intensity = 100000
else:
    intensity = 1
# Made a point in centre of 2D array
point = np.zeros((xy_size, xy_size, laserPSF.shape[2]))
# point[25, 25, 1] = intensity
# point[75, 75, -1] = intensity
point[laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = intensity
# point[laserPSF.shape[0]//2, laserPSF.shape[1]//2+20, laserPSF.shape[2] // 2] = intensity

## Spherical ground truth  ##
# radius = 40
# sphere_centre = (point.shape[0]//2, point.shape[1]//2, point.shape[2] // 2)
# point = confmain.emptysphere3D(point, radius, sphere_centre) * intensity
## More Complex Spherical Ground Truth ##
# sample = point
# sample = confmain.emptysphere3D(sample, int(sample.shape[0]*0.4), (sample.shape[1]//2, sample.shape[0]//2, sample.shape[2]//2))
# sample2 = confmain.emptysphere3D(sample, int(sample.shape[0]*0.25), (sample.shape[1]//2.5, sample.shape[0]//2.5, sample.shape[2]//2.5))
# sample3 = confmain.emptysphere3D(sample, int(sample.shape[0]*0.05), (sample.shape[1]//1.4, sample.shape[0]//1.4, sample.shape[2]//1.7))
# sample4 = confmain.emptysphere3D(sample, int(sample.shape[0]*0.05), (sample.shape[1]//2.5, sample.shape[0]//1.4, sample.shape[2]//1.4))
# point = (sample+sample2+sample3+sample4) * intensity


### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
# Establish the PSF of the emission wwavelength as the light passes through the lens after scanning
emission_PSF = confmain.radial_PSF(xy_size, pixel_size, stack_size, emission_wavelength)
emission_PSF = np.moveaxis(emission_PSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.
emission_PSF = (emission_PSF / emission_PSF.sum())  # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)
# Takes the psf and sample arrays as inputs and scans across them.
# Returns an array at the same xy size with a z depth of x*y.
if mode == "Widefield":
    sums = signal.fftconvolve(point, emission_PSF, "same")
    for x in range(0, sums.shape[1]):
        for y in range(0,sums.shape[0]):
            for z in range(0, sums.shape[2]):
                if sums[y,x,z] < 0:
                    sums[y,x,z] = 0
    if include_shot_noise == "Y":
        shot_noise = confmain.shot_noise(np.sqrt(sums), sums, fix_seed)
        print("Shot noise added")
    else:
        shot_noise = 0
        print("Shot noise not added")
    sums = sums + shot_noise
else:
    sums = confmain.stage_scanning(laserPSF, point, emission_PSF, include_shot_noise, fix_seed)


### CAMERA SETUP ###
# Camera sensor, based on optical magnification and pixel size.
camerapixel_per_groundpixel = camera_pixel_size / pixel_size

# Used to determine the number of the ground pixels that exist within each bin
# We use the mag_ratio to step across the array and select out regions that are multiples of it out.
mag_ratio = camerapixel_per_groundpixel / magnification
print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)


##### IMAGING TIME #####
### UPSCALING ###
# If the mag_ratio is not a whole number then we have an issue with distributing the values to the new array accurately.
# The below functions aims to upscale each 2D scan to allow for an integer based binning method.

upscale = 0         # a simple counter to determine if upscaling has been done or not.
upscaled_sum = 0
upscale_mag_ratio = 0
if Fraction(mag_ratio-int(mag_ratio)) != 0:
    upscaled_sum, upscale_mag_ratio = confmain.upscale(sums, mag_ratio)
    upscale = 1


### BINNING ###
# Initialise an empty array, with a size calculated by the above ratios.
# Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into an end pixel.
# Bins each image collected at the detector to the camera pixel size.
binned_image = np.zeros((int(sums.shape[0] // mag_ratio),
                         int(sums.shape[1] // mag_ratio),
                         int(sums.shape[2])))
print("Binning Data to Camera")

# If upscaling has occured we need to use different variables for the function.
sums_list = [sums, upscaled_sum]
mag_ratio_list = [mag_ratio, upscale_mag_ratio]

# Actual binning step.
binned_image = confmain.binning(sums_list[upscale], binned_image, mag_ratio_list[upscale])
print("Data Binned")


### QUANTUM EFFICIENCY ###
QE_image = binned_image * QE
print("QE step")


### NOISE ###
print("Creating noise.")
if include_read_noise == "Y":
    read_noise = confmain.read_noise(QE_image, read_mean, read_std)
    print("Read noise generated.")
else:
    read_noise = 0
# Fix the seed for fixed pattern noise
if include_fixed_pattern_noise == "Y":
    np.random.seed(100)
    fixed_pattern_noise = np.random.normal(1, fixed_pattern_deviation, (QE_image.shape[0],
                                                                        QE_image.shape[1],
                                                                        QE_image.shape[2]))
else:
    fixed_pattern_noise = 1
print("Fixed Pattern noise generated.")


# Sum the noises and the image
noisy_image = (QE_image + read_noise) * fixed_pattern_noise
print("Combining noises with image. Complete!")


### GAIN, COUNT AND INTEGER ###
# Multiply by gain to convert from successful incidence photons and noise to electrons.
print("All about them gains.")
camera_gain = noisy_image * gain

# 100 count added as this is what camera's do.
print("Count on it.")
camera_plusCount = camera_gain + count

# Convert to integer as a camera output can only take integers
# Conversion to: USER INT VALUE 16
print("Integerising at the detector")
camera_view = camera_plusCount.astype(np.uint16)


### PINHOLE ###
# Multiply by pinhole, which is centered and offset.
# Produces a simple circle mask centred at x,y.
if mode == "Widefield":
    pinhole_sum = noisy_image
else:
    print("Time to add our digital pinhole.")
    circle_pinhole = proc.circle_mask(camera_view, pinhole_radius,
                                      (camera_view.shape[1] // 2 + offset, camera_view.shape[0] // 2 + offset))
    # Produce an empty array to place the results into.
    pinhole_sum = np.zeros((camera_view.shape[0], camera_view.shape[1], camera_view.shape[2]))

    # Multiplies the binned image with the circle pinhole at each z position.
    # This digitally adds our pinhole to the binned image at the size we desire.
    for i in range(0, camera_view.shape[2]):
        pinhole_sum[:, :, i] = camera_view[:, :, i] * circle_pinhole
    print("Pinholes added.")


### CONFOCAL SUMMING ###
# Reconstruct the image based on the collected arrays.
if mode == "Widefield":
    conf_array = np.sum(pinhole_sum,2)
    print("Widefield Image, Check")
elif mode == "Confocal":
    print("So it's CONFOCAL imaging time.")
    conf_array = confmain.confocal(pinhole_sum, point)
    print("CONFOCAL, DEPLOY IMAGE!!")

elif mode == "ISM":
    print("ISM, a wise choice.")
    conf_array = confmain.ISM(pinhole_sum, point, pinhole_radius)
    print("ISM, I See More? Check the image to find out.")

else:
    # Will become redundant once checks are in place.
    print("You didn't select an appropriate reconstruction form...")
    print("To save you some time we are outputting the Confocal based reconstruction of the image.")
    conf_array = confmain.confocal(pinhole_sum, point)


### SAVE, PREVIEW ###
if Preview == "Y":
    plt.imshow(conf_array)
    plt.show()

if SAVE == "Y":
    print("Saving Image")
    proc.savetiff(filename+".tif", conf_array)
    print("Image saved.")



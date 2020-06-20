import numpy as np
import Confocal_Samurai as sam
import matplotlib.pyplot as plt
import microscPSF as msPSF
from scipy import signal
from fractions import Fraction

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
laser_power = 100       # Laser power in ????DADSA?
# PSF
# NA = 1.4                # Numerical aperture
# wavelength = 0.600      # Wavelength in microns
# PINHOLE #
pinhole_radius = 1        # Radius of the pinhole in pixels.
offset = 1                # Offsets the pinhole. Doesnt really do much at this stage.
# CAMERA
camera_pixel_size = 6   # Camera pixel size in microns. usual sizes = 6 microns or 11 microns
magnification = 200     # Lens magnification
QE = 0.7                # Quantum Efficiency
# gain = 2                # Camera gain. Usually 2 per incidence photon
# NOISE
# read_mean = 2           # Read noise mean level
# read_std = 2             # Read noise standard deviation level
# fixed_pattern_deviation = 0.001  # Fixed pattern standard deviation. usually affects 0.1% of pixels.
# MODE #
mode = "confocal"       # Mode refers to whether we are doing confocal or ISM imaging.
# SAVE
# Preview = "N"
# SAVE = "N"  # Save parameter, input Y to save, other parameters will not save.
# filename = "X"

###### MAIN PROGRAM ######
# Order of the program: produce sample and PSF, multiply, convolute and sum them in z for each xy point, bin the images
#                       as though they have been magnified to the cameras pixels, then process the data through
#                       confocal or ISM methods to produce the final image.
### PSF Generation ###
# Made a 3D PSF
# Each pixel = 5 nanometres.
# Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
laserPSF = sam.radial_PSF(xy_size, pixel_size, stack_size)
laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

laserPSF = laserPSF / laserPSF.sum()      # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)


### SAMPLE PARAMETERS ###
# Made a point in centre of 2D array
point = np.zeros((xy_size, xy_size, laserPSF.shape[2]))
point[25, 25, 1] = 1
point[75, 75, -1] = 1
point[laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1



### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
# Takes the psf and sample arrays as inputs and scans across them.
# Returns an array at the same xy size with a z depth of x*y.
sums = sam.stage_scanning(laserPSF, point)


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
    upscaled_sum, upscale_mag_ratio = sam.upscale(sums, mag_ratio)
    upscale = 1


### BINNING ###
# Initialise an empty array, with a size calculated by the above ratios.
# Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into an end pixel.
# Bins each image collected at the detector to the camera pixel size.
binned_image = np.zeros((int(sums.shape[0] // mag_ratio),
                         int(sums.shape[1] // mag_ratio),
                         int(sums.shape[2])))
print("Binning Data to Camera")

# if upscaling has occured we need to use different variables for the function.
sums_list = [sums, upscaled_sum]
mag_ratio_list = [mag_ratio, upscale_mag_ratio]

binned_image = sam.binning(sums_list[upscale], binned_image,mag_ratio_list[upscale])

print("Data Binned")


# Multiply by pinhole, which is centered and offset.
# Produces a simple circle mask centred at x,y.
circle_pinhole = sam.circle_mask(binned_image, pinhole_radius,
                                 (binned_image.shape[1] // 2 + offset, binned_image.shape[0] // 2 + offset))
# Produce an empty array to place the results into.
pinhole_sum = np.zeros((binned_image.shape[0], binned_image.shape[1], binned_image.shape[2]))

# Multiplies the binned image with the circle pinhole at each z position.
# This digitally adds our pinhole to the binned image at the size we desire.
for i in range(0, binned_image.shape[2]):
    pinhole_sum[:, :, i] = binned_image[:, :, i] * circle_pinhole


# Account for Quantum efficiency.
# camera_image = pinhole_sum * QE
# print("QE step")

### Need to add noise, gain etc... here.

### CONFOCAL SUMMING ### #### FIX THIS .. ISSUES WITH SPACIAL COMPRESSION IN Z.
# Reconstruct the image based on the collected arrays.
if mode == "confocal":
    # Set up the confocal array
    conf_array = np.zeros((point.shape[0], point.shape[1]))

    # Iterate through the z stack and sum the values and add them to the appropriate place on the image.
    for i in range(0, pinhole_sum.shape[2]):
        conf_array[i%point.shape[0], i//point.shape[1]] = np.sum(pinhole_sum[:,:,i])
        print(i//point.shape[1], i%point.shape[0])

# elif mode == "ISM":
    ##ƒå˚ø˙^¨¨å¨
    ### BUILD ME ###
    # sums = sums


plt.imshow(conf_array)
plt.show()

# plt.imshow(pinhole)
# flat = np.sum(pinhole, axis=2)
### PLOTTING ###
# plt.imshow(flat)
# position = 100
# plt.subplot(141)
# plt.imshow(point[:,:,2])
# plt.title("Point")
# plt.subplot(142)
# plt.imshow(radPSF[:,:,position])
# plt.title("Radial PSF")
# plt.subplot(143)
# plt.imshow(scan[:,:,position])
# plt.title("Scanned")
# plt.subplot(144)
# plt.imshow(pinhole[:,:,position])
# plt.title("Pinhole")
# plt.text(-150,-30,"Frame " + str(position) + " of " + str(pinhole.shape[2]))
# plt.show()

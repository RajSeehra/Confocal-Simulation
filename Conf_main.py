import numpy as np
import Confocal_Samurai as sam
import matplotlib.pyplot as plt
import microscPSF as msPSF
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
stack_size = 40         # Z depth for the PSF
laser_power = 100       # Laser power in ????DADSA?
# PSF
# NA = 1.4                # Numerical aperture
# wavelength = 0.600      # Wavelength in microns
# PINHOLE #
offset = 1                # Offsets the pinhole. Doesnt really do much at this stage.
# CAMERA
camera_pixel_size = 6   # Camera pixel size in microns. usual sizes = 6 microns or 11 microns
magnification = 100     # Lens magnification
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


### PSF Generation ###
# Made a 3D PSF
# Each pixel = 5 nanometres.
# Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
laserPSF = sam.radial_PSF(xy_size, pixel_size, stack_size)
laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

laserPSF = laserPSF / laserPSF.sum()          # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)


### SAMPLE PARAMETERS ###
# Made a point in centre of 2D array
point = np.zeros((xy_size, xy_size, laserPSF.shape[2]))
point[25, 25, laserPSF.shape[2] // 2 + 19] = 1
point[75, 75, laserPSF.shape[2] // 2 -19] = 1
point[laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1


### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))
laser_illum = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))

# Produce an array that will receive the 2D samples we collect.
sums = np.zeros((point.shape[1], point.shape[0], int(point.shape[1] * point.shape[0])+1))
counter = 0


# Iterates across the array to produce arrays illuminated by the sample, with a laser blurred by the first lens.
for x in range(0, point.shape[1]):
    for y in range(0, point.shape[0]):
        # Multiplies the PSF multiplied laser with the sample. The centre of the sample is moved to position x,y
        # on the laser array, as this is a stage scanning microscope.
        laser_illum = sam.array_multiply(laserPSF, point, x, y)
        # Convolute the produced array with the PSF to simulate the second lens.
        for i in range(0, point.shape[2]):
            scan[:,:,i] = np.rot90(signal.fftconvolve(laserPSF[:,:,i], laser_illum[:,:,i], mode="same"),2)
            # scan[:,:,i] = np.rot90(sam.kernel_filter_2D(laserPSF[:, :, i], laser_illum[:, :, i]), 2)        # When running a larger image over a smaller one it rotates the resulting info.
        print(x,y)
        # Flatten and sum z stack.
        z_sum = np.sum(scan, 2)

        # Multiply by pinhole, which is centered and offset.
        circle_pinhole = sam.circle_mask(scan, 1, (point.shape[1]//2 + offset, point.shape[0]//2 + offset))  # Produces a simple circle mask centred at x,y.
        pinhole_sum = z_sum * circle_pinhole

        # Add to the collection array
        sums[:,:,counter] = pinhole_sum
        print("Counter: ", counter)
        counter = counter + 1

### CONFOCAL SUMMING ###
# Reconstruct the image based on the collected arrays.
if mode == "confocal":
    conf_array= np.zeros((100,100))

    for i in range(0, sums.shape[2]-1):
        conf_array[i%100, i//100] = np.sum(sums[:,:,i])
        print(i//100, i%100)
elif mode == "ISM":
    ###ƒå˚ø˙^¨¨å¨
    ##  BUILD ME ###
    sums = sums


### CAMERA SETUP ###
# Camera sensor, based on optical magnification and pixel size.
camerapixel_per_groundpixel = (camera_pixel_size) / pixel_size

# Used to determine the number of the ground pixels that exist within each bin
mag_ratio = camerapixel_per_groundpixel / magnification
print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)


### IMAGING TIME ###
# Initialise an empty array, with a size calculated by the above ratios.
# Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into a pixel.
camera_image = np.zeros((int(conf_array.shape[0] // mag_ratio), int(conf_array.shape[1] // mag_ratio)))

# Iterate each position in the array and average the pixels in the range from the diffraction limited image.
# We use the mag_ratio to step across the array and select out regions that are multiples of it out.
for y in range(0, camera_image.shape[0]):
    for x in range(0, camera_image.shape[1]):
        pixel_section = conf_array[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
                                   int(x * mag_ratio):int(x * mag_ratio + mag_ratio)]
        camera_image[y, x] = np.sum(pixel_section)  # Take the mean value of the section and bin it to the camera.
print("Collecting Data")

# Account for Quantum efficiency.
camera_image = camera_image * QE
print("QE step")

plt.imshow(camera_image)
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
plt.show()

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
stack_size = 100         # Z depth for the PSF
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
point[25, 25, 1] = 1
point[75, 75, -1] = 1
point[laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1


### MAIN PROGRAM ###
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

        # Add to the collection array
        sums[:,:,counter] = z_sum
        print("Counter: ", counter)
        counter = counter + 1


### CAMERA SETUP ###
# Camera sensor, based on optical magnification and pixel size.
camerapixel_per_groundpixel = camera_pixel_size / pixel_size

# Used to determine the number of the ground pixels that exist within each bin
mag_ratio = camerapixel_per_groundpixel / magnification
print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)


### IMAGING TIME ###
# Initialise an empty array, with a size calculated by the above ratios.
# Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into an end pixel.
binned_image = np.zeros((int(sums.shape[0] // mag_ratio),
                         int(sums.shape[1] // mag_ratio),
                         int(sums.shape[2])))

# Iterate each position in the array and average the pixels in the range from the diffraction limited image.
# We use the mag_ratio to step across the array and select out regions that are multiples of it out.
for z in range(0, binned_image.shape[2]):
    for y in range(0, binned_image.shape[0]):
        for x in range(0, binned_image.shape[1]):
            # Takes the convoluted and summed data and bins the sections into a new image
            pixel_section = sums[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
                            int(x * mag_ratio):int(x * mag_ratio + mag_ratio),
                            z]
            binned_image[y, x, z] = np.sum(pixel_section) # Take the sum value of the section and bin it to the camera.
            print(x,y,z)

print("Binning Data to Camera")




# Multiply by pinhole, which is centered and offset.
# Produces a simple circle mask centred at x,y.
circle_pinhole = sam.circle_mask(binned_image, 1, (binned_image.shape[1] // 2 + offset, binned_image.shape[0] // 2 + offset))
# Produce an empty array to place the results into.
pinhole_sum = np.zeros((binned_image.shape[0], binned_image.shape[1], binned_image.shape[2]))

# Multiplies the binned image with the circle pinhole at each z position
for i in range(0, binned_image.shape[2]):
    pinhole_sum[:, :, i] = binned_image[:, :, i] * circle_pinhole




# Account for Quantum efficiency.
camera_image = pinhole_sum * QE
print("QE step")



### CONFOCAL SUMMING ### #### FIX THIS .. ISSUES WITH SPACIAL COMPRESSION IN Z.
# Reconstruct the image based on the collected arrays.
if mode == "confocal":
    # Set up the confocal array and the intermediate array.
    # The intermediate acts as a holding cell for the array values once summed up. They are returned to their original
    # position to make the next step easier.
    conf_array = np.zeros((point.shape[0], point.shape[1]))
    # intermediate = np.zeroes((xy_size,xy_size))
    #
    # # Initially we sum and organise the arrays as in the original image size. This allows us to make a pseudo-replica
    # # of the original image which we then bin into the correct positions
    # for i in range(0, pinhole_sum.shape[2] - 1):
    #     intermediate[i % pinhole_sum.shape[0], i // pinhole_sum.shape[1]] = pinhole_sum[:, :, i]
    #     print(i // pinhole_sum.shape[1], i % pinhole_sum.shape[0])
    #
    # # If the mag_ratio is a float variable then the array will stumble and take steps of different size.
    # # Hence in such cases we adjust and add on the excess to the right/bottom
    # # and subtract it from the left/top as needed.
    #
    # if mag_ratio - int(mag_ratio) > 0:
    #     remainder = mag_ratio - int(mag_ratio)
    #
    #     for y in range(0, conf_array.shape[0]):
    #         for x in range(0, conf_array.shape[1]):
    #             conf_array[y, x] = np.sum(pinhole_sum[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
    #                                                   int(x * mag_ratio):int(x * mag_ratio + mag_ratio)])
    #
    #             # These next variables collect the next x and y row/column of the same length as the mag array
    #             array_remainder_top = np.sum(pinhole_sum[int(y * mag_ratio -1),
    #                                             int(x * mag_ratio):int(x * mag_ratio + mag_ratio)])
    #             if y * mag_ratio - 1 < 0:
    #                 array_remainder_top = 0
    #
    #             array_remainder_bottom = np.sum(pinhole_sum[int(y * mag_ratio + mag_ratio + 1),
    #                                             int(x * mag_ratio):int(x * mag_ratio + mag_ratio)])
    #             if y * mag_ratio + mag_ratio + 1 > 0:
    #                 array_remainder_bottom = 0
    #
    #             array_remainder_left = np.sum(pinhole_sum[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
    #                                                      int(x * mag_ratio - 1)])
    #             array_remainder_right = np.sum(pinhole_sum[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
    #                                             int(x * mag_ratio + mag_ratio + 1)])
    #             array_remainder_bottom_right = np.sum(pinhole_sum[int(y * mag_ratio + mag_ratio + 1),
    #                                                  int(x * mag_ratio + mag_ratio + 1)])
    #
    #             #
    #
    #             conf_array[y, x] = conf_array[y, x] +
    # elif mag_ratio - int(mag_ratio) == 0:
    for i in range(0, pinhole_sum.shape[2]-1):
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

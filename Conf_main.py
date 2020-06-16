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
#  Laser PSF #
xy_size = 100
pixel_size = 0.005
stack_size = 40


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
# point[50, 50, laserPSF.shape[2] // 2 + 8] = 1
# point[100, 100, laserPSF.shape[2] // 2 + 4] = 1
point[00:laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1


### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))
laser_illum = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))

# Produce an array that will receive the 2D samples we collect.
sums = np.zeros((point.shape[1], point.shape[0], int(point.shape[1] * point.shape[0])+1))
counter = 0


# Iterates across the array to produce arrays illuminated by the sample, with a laser blurred by the first lens.
for x in range(0, point.shape[1]):
    for y in range(0, point.shape[0]):
        laser_illum = sam.array_multiply(point, laserPSF, x, y)
        # Convolute the produced array with the PSF to simulate the second lens.
        for i in range(0, point.shape[2]):
            scan[:,:,i] = signal.fftconvolve(laserPSF[:,:,i], laser_illum[:,:,i], mode="same")
            # scan[:,:,i] = np.rot90(sam.kernel_filter_2D(laserPSF[:, :, i], laser_illum[:, :, i]), 2)        # When running a larger image over a smaller one it rotates the resulting info.
        print(x,y)
        # Flatten and sum z stack.
        z_sum = np.sum(scan, 2)

        # Multiply by pinhole
        circle_pinhole = 1 #sam.circle_mask(scan, 10, (x, y))  # Produces a simple circle mask centred at x,y.
        pinhole_sum = z_sum * circle_pinhole

        # Add to the collection array
        sums[:,:,counter] = pinhole_sum
        print("Counter: ", counter)
        counter = counter + 1


# REMAKE IN PROGRESS##˙^¨¨©˙åø¨¨©ƒø©å


### SPACIAL FILTER PARAMETERS ###
# 1/0 pinhole to make.!!!---> with mobility relative to the centre. circle.
# circle_pinhole = sam.circle_mask(scan, 20, (200, 200))  # Shifting the centre, offsets the image by the same value.

# # Made a gaussian. (Our spacial filter)  NEED SCALE FOR THIS PINHOLE AS IT WILL VASTLY IMPACT QUALITY.
# spacial_filter = sam.Gaussian_Map((scan.shape[1], scan.shape[0]), 0, 0, 0, 1, 1)
# spacial_filter[spacial_filter < 0.05] = 0
# spacial_filter = spacial_filter / np.sum(spacial_filter)

### PINHOLE ###
# pinhole = np.zeros((scan.shape[1], scan.shape[0], scan.shape[2]))
# Each array in the 3D scan is convoluted with the spacial filter individually as they are each snapshots in space.
# pinhole = np.rot90(sam.kernel_filter_2D(circle_pinhole, scan_sum), 2)
# pinhole = sam.kernel_filter_2D(spacial_filter, scan_sum)
# for i in range(0, scan.shape[2]):
#     pinhole[:, :, i] = np.rot90(sam.kernel_filter_2D(circle_pinhole, scan[:, :, i]), 2)

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

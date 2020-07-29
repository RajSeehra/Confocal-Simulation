import numpy as np
import Confocal_Samurai as sam
import matplotlib.pyplot as plt
import microscPSF as msPSF
from scipy import signal


# This program takes a 2D sample and simulates a stage scanning microscope. Hence the sample is 'moved'
# relative to the detector.
# The final output is a 3D image whereby the sample has passed through: a lens as denoted by a convolution with a
# radial PSF, and a pinhole simulated by a layer wise convolution with a gaussian spacial filter.
# Array sizes need to be odd to work with the convolution as a centre point needs to be determinable.


### SAMPLE PARAMETERS ###
# Made a point in centre of 2D array
point = np.zeros((51, 51))
point[25, 25] = 1
point[15, 15] = 1

### PSF Generation ###
# Made a 3D PSF
# Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 suffices and is hella faster.
radPSF = sam.radial_PSF(101, 0.05)
radPSF = np.moveaxis(radPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
scan = np.zeros((radPSF.shape[1], radPSF.shape[0], radPSF.shape[2]))
scan = np.rot90(sam.kernel_filter(radPSF, point), 2)        # When running a larger image over a smaller one it rotates the resulting info.

### DEBUGGING: CHECK FOR FLATNESS ###
# Check for flatness in single particle.
# check = scan[:,:,100]-radPSF[:,:,100]
# print("minimum: " + str(np.min(check)), "maximum: " + str(np.max(check)))

### SPACIAL FILTER PARAMETERS ###
# Made a gaussian. (Our spacial filter)  NEED SCALE FOR THIS PINHOLE AS IT WILL VASTLY IMPACT QUALITY.
spacial_filter = sam.Gaussian_Map((scan.shape[1], scan.shape[0]), 0, 0, 0, 1, 1)

### PINHOLE ###
pinhole = np.zeros((scan.shape[1], scan.shape[0], scan.shape[2]))
# Each array in the 3D scan is convoluted with the spacial filter individually as they are each snapshots in space.
for i in range(0, scan.shape[2]):
    pinhole[:, :, i] = np.rot90(sam.kernel_filter(spacial_filter, scan[:, :, i]), 2)


### PLOTTING ###
position = 100
plt.subplot(141)
plt.imshow(point[:,:])
plt.title("Point")
plt.subplot(142)
plt.imshow(radPSF[:,:,position])
plt.title("Radial PSF")
plt.subplot(143)
plt.imshow(scan[:,:,position])
plt.title("Scanned")
plt.subplot(144)
plt.imshow(pinhole[:,:,position])
plt.title("Pinhole")
plt.text(-180,-40,"Frame " + str(position) + " of " + str(pinhole.shape[2]))
plt.show()

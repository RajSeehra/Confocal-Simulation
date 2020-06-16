import numpy as np
import matplotlib.pyplot as plt
import Confocal_Samurai as sam
from scipy import signal
from PIL import Image

# centre_x = 400
# centre_y = 500
# a_x = -centre_x  # distance of left edge of screen relative to centre x.
# b_x = 1000 - centre_x  # distance of right edge of screen relative to centre x.
# a_y = -centre_y  # distance of top edge of screen relative to centre y.
# b_y = 1000 - centre_y  # distance of bottom edge of screen relative to centre y.
#
# r = 100
# # Produce circle mask, ones grid = to original file and cut out.
# y, x = np.ogrid[a_x:b_x, a_y:b_y]  # produces a list which collapses to 0 at the centre in x and y
# mask = x * x + y * y <= r * r  #
# ones = np.ones((1000, 1000))
# ones[mask] = 0



# x = np.load("conf2.npy")
# print(np.max(x))
# sam.savetiffs("output.tif", x)
# array= np.zeros((201,201))
#
# for i in range(0, x.shape[2]):
#     array[i%201, i//201] = np.sum(x[:,:,i])
#     print(i//201, i%201)
#
# plt.imshow(array)
# plt.imshow(x[:,:,10000])
# plt.show()





### INPUTS ###
#  Laser PSF #
xy_size = 201
pixel_size = 0.005
stack_size = 100


### PSF Generation ###
# Made a 3D PSF
# Each pixel = 5 nanometres.
# Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
laserPSF = sam.radial_PSF(xy_size, pixel_size, stack_size)
laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

laserPSF = laserPSF / laserPSF.sum()          # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)


### SAMPLE PARAMETERS ###
# Made a point in centre of 2D array
point = np.zeros((201, 201, laserPSF.shape[2]))
point[50, 50, laserPSF.shape[2] // 2 + 8] = 1
point[100, 100, laserPSF.shape[2] // 2 + 4] = 1
point[00:100, 100, laserPSF.shape[2] // 2] = 1

a = sam.array_multiply(point, laserPSF, 125, 125)
#
# scan = scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))
#
#
# for i in range(0, point.shape[2]):
#     scan[:,:,i] = signal.fftconvolve(laserPSF[:, :, i], a[:, :, i], mode="same")
#
# # Multiply by pinhole
# summer = np.sum(scan,2)
# circle_pinhole = sam.circle_mask(scan, 20, (125, 125))  # Produces a simple circle mask centred at x,y.
# pinhole_sum = summer * circle_pinhole
#
# plt.subplot(121)
# plt.imshow(summer)
# plt.subplot(122)
# plt.imshow(pinhole_sum)
# plt.show()


plt.imshow(a[:,:,50])
plt.show()


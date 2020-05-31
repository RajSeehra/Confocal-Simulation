import numpy as np
import matplotlib.pyplot as plt
import pywt
import Confocal_Samurai as sam


# Generates randompoints in a 2D array.
def randompoints(size):
    # Random seed generator
    np.random.seed()
    x = np.random.randint(0, 101, size)
    mapper = pywt.threshold(x, 100, substitute=0, mode='greater')
    return mapper


# Build the 2D random point array. Needs to be odd numbered array to be convoluted.
mapper = randompoints([101, 101])
# Builds the gaussian laser which is passing over our point.
mapper2 = sam.Gaussian_Map([100, 100], 0, 0, 0, 1.5, 1000)

process = sam.kernel_filter(mapper2, mapper)
process = np.rot90(process, 2)

plt.subplot(131)
plt.imshow(mapper)
plt.title("Random Points")
plt.subplot(132)
plt.imshow(mapper2)
plt.title("Gaussian")
plt.subplot(133)
plt.imshow(process)
plt.title('Gauss')
plt.show()

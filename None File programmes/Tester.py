import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
import Confocal_Samurai as sam
from scipy import signal
from PIL import Image
import microscPSF as msc
from multiprocessing import pool
from scipy.stats import binned_statistic_2d
from skimage.transform import resize

# psf = sam.radial_PSF(100, 0.02, 100)
# laserPSF = np.moveaxis(psf, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.
#
# plt.imshow(laserPSF[:,:,1])
# plt.show()
# cProfile.run
# with multipro

# data = np.asarray(np.random.normal(2,2,(100,100,10000)))
# sums = data
# mag_ratio = 1.5
# binned_image = np.zeros((int(sums.shape[0] // mag_ratio),
#                          int(sums.shape[1] // mag_ratio),
#                          int(sums.shape[2])))
# sam.binning(sums, binned_image, mag_ratio)

# np.ra

def dict_datacheck_lesser_greater(data, lesser_values, greater_values):
    ''' Will take a dictionary of variables and check if the check variable is in the ranges prescribed by the function.

    :param data:
        The dictionary containing the variables to be checked.
    :param lesser_value:
        The minimum value for the variable in question.
    :param greater_value:
        The maximum value for the variable.
    :return:
    '''
    check_variable_names = [key for key in data.keys()]

    for i in range(0, len(check_variable_names)):
        check_variable_name = check_variable_names[i]
        test_variable = float(data[check_variable_name])

        lesser_value = float(lesser_values[i])
        greater_value = float(greater_values[i])

        while test_variable < lesser_value or test_variable > greater_value:
            data[check_variable_name] = input(check_variable_name + " is not in the range expected:" + str(lesser_value) + "-" + str(greater_value))
            try:
                test_variable = float(data[check_variable_name])

            except:
                print(check_variable_name + " input error. Try again.")


def simple_datacheck_lesser_greater(variable, variable_name, lesser_value, greater_value):
    ''' Will take a variable and check if the variable is in the ranges required.

    :param variable:
        The variable to be tested containing the variables to be checked.
    :param variable_name:
        The name of the variable to be tested. Mainly used in the print function.
    :param lesser_value:
        The minimum value for the variable in question.
    :param greater_value:
        The maximum value for the variable.
    :return: test_data -
        The value to be returned that is now in the range required.
    '''
    test_data = variable

    while test_data < lesser_value or test_data > greater_value:
        test_data = input(variable_name + " is not in the range expected:" + str(lesser_value) + "-" + str(greater_value))
        try:
            test_data = float(test_data)

        except:
            print(variable_name + " input error. Try again.")
            test_data = lesser_value - 1

    return test_data


def simple_datacheck_string(variable, variable_name, string_options):
    ''' Will take a variable and check if the variable is one of the options mentioned.

    :param variable:
        The variable to be tested containing the variables to be checked.
    :param variable_name:
        The name of the variable to be tested. Mainly used in the print function.
    :param string_options:
        A list of the possible options that the variable could be.
    :return: test_data -
        The value to be returned that is now in the range required.
    '''
    test_data = variable

    while test_data not in string_options:
        test_data = input(variable_name + " is not of the predefined variables:" + str(string_options))
        test_data = str(test_data)

    return test_data

mode = -0
cheese = 200
string_mode = ["Confocal", "ISM"]
# mode = simple_datacheck_string(mode,'mode', string_mode)
# print (mode)
mode = simple_datacheck_lesser_greater(mode, "mode", -10, np.inf)

# data = dict(mode=mode, cheese=cheese)
# print([key for key in data.keys()][0])
# key = [key for key in data.keys()]
# lesser_values = [-10,20]
# greater_values = [40, 100]
#
# print(data[key[0]])
# datacheck_lesser_greater(data, lesser_values, greater_values)


# point = np.zeros((100, 100, 100))
# radius = 40
# sphere_centre = (point.shape[0]//2, point.shape[1]//2, point.shape[2] // 2)
# conf_array = sam.emptysphere3D(point, radius, sphere_centre)
#
# plt.imshow(conf_array[:,:,50])
# plt.show()
#
# def emptysphere3D(array, radius, sphere_centre):
#     """"" A function that takes an array size, sphere centre and radius and constructs an empty sphere.
#     Parameters
#     ----------
#     array : arraylike
#         the size of the array you wish for the sphere to be held in.
#     radius : int
#         the radius of the sphere. uniform in x, y, and z.
#     sphere_centre : tuple
#         the centre coordinates of the sphere.
#
#     """
#     # PARAMETERS FOR CIRCLE MASK
#     centre_x = sphere_centre[0]
#     centre_y = sphere_centre[1]
#     centre_z = sphere_centre[2]
#     a_x = -centre_x                         # distance of left edge of screen relative to centre x.
#     b_x = array.shape[1] - centre_x         # distance of right edge of screen relative to centre x.
#     a_y = -centre_y                         # distance of top edge of screen relative to centre y.
#     b_y = array.shape[0] - centre_y         # distance of bottom edge of screen relative to centre y.
#     a_z = - centre_z                        # distance of z edge of screen relative to centre z.
#     b_z = array.shape[2] - centre_z         # distance of z edge of screen relative to centre x.
#
#     r = radius
#     # Produce circle mask, ones grid = to original file and cut out.
#     y, x, z = np.ogrid[a_y:b_y, a_x:b_x, a_z:b_z]       # produces a list which collapses to 0 at the centre in x and y
#     mask1 = x*x + y*y + z*z <= r*r                      # produces a true/false array where the centre is true.
#     mask2 = x*x + y*y + z*z - radius*3 <= r*r           # produces a second mask which helps construct our edge.
#     ones= np.zeros((array.shape[1], array.shape[0], array.shape[2]))
#     # combine the two mask to produce a sphere and then cut out the centre.
#     ones[mask2] = 1                     # uses the mask to turn the zeroes to ones in the TRUE zone of mask.
#     ones[mask1] = 0                     # uses the mask to turn the ones to zeroes in the TRUE zone of mask.
#
#     return ones
#
# array = np.zeros((100,100,10000))
# radius = 40
# centre_xyz = (50,50,5000)
# a = emptysphere3D(array, radius, centre_xyz)
# plt.imshow(a[:,:,5000])
# # plt.imshow(mask1[:,:,5000])
# # plt.imshow(mask2[:,:,5000])
# plt.show()
#
#

# scaling_array = np.zeros((100,100, 10000))
# for z in range(0, 10000):
#     x = z//100
#     y = z%100
#     print(x%mag_ratio,y%mag_ratio)
#
#     if x%mag_ratio == 0 and y%mag_ratio ==0:
#         scaling_array[int(np.floor(y)):int(np.ceil(y+mag_ratio)),
#                     int(np.floor(x)):int(np.ceil(x+mag_ratio)),z] = 1
#
#     elif x%mag_ratio != 0:
#         x = x - (mag_ratio-int(mag_ratio))
#         scaling_array[int(np.floor(y)):int(np.ceil(y + mag_ratio)),
#                       int(np.floor(x)):int(np.ceil(x + mag_ratio)), z] = 1
#
#     elif y%mag_ratio != 0:
#         y = y - (mag_ratio - int(mag_ratio))
#         scaling_array[int(np.floor(y)):int(np.ceil(y + mag_ratio)),
#         int(np.floor(x)):int(np.ceil(x + mag_ratio)), z] = 1
#
#     else:
#         x = x - (mag_ratio-int(mag_ratio))
#         y = y - (mag_ratio-int(mag_ratio))
#         scaling_array[int(np.floor(y)):int(np.ceil(y + mag_ratio)),
#                       int(np.floor(x)):int(np.ceil(x + mag_ratio)), z] = 1
#
#
#
# scale = np.sum(scaling_array)
#
#
#
#
# plt.subplot(121)
# plt.imshow(data[:,:,0])
# plt.subplot(122)
# plt.imshow(scaling_array[:,:,0])
# plt.show()
    # scaling_array[int(np.floor(yedge[int(y)])):int(np.ceil(yedge[int(y+1)])),
    #               int(np.floor(xedge[int(x)])):int(np.ceil(xedge[int(x+1)])),z] = 1
#             print("X",x)
#         print("Y",y)

# z = 4
# scaling_array[int(np.floor(yedge[z])):int(np.ceil(yedge[z+1])), int(np.floor(xedge[z])):int(np.ceil(xedge[z+1])),0] = 1
# scaling_array[int(np.floor(yedge[z])):int(np.ceil(yedge[z+1])), int(np.floor(xedge[z])):int(np.ceil(xedge[z+1])),0] = 1
# print(int(np.floor(yedge[z])), int(np.ceil(yedge[z+1])), int(np.floor(xedge[z])), int(np.ceil(xedge[z+1])))
            # 0,0 ->1,1,->2,2
            # scaling_array[y:int(y+mag_ratio+1),x:int(x+mag_ratio+1),z] = 1
            # scaling_array[int(y+mag_ratio+1), int(x+mag_ratio+1),z] = 1
            # mask = x * x + y * y <= r * r  # produces a true/false array where the centre is true.
            # int(xedge[0]):int(xedge[1]+1)



# for z in range(0, binned_image.shape[2]):
#     for y in range(0, binned_image.shape[0]):
#         for x in range(0, binned_image.shape[1]):
#             # Takes the convoluted and summed data and bins the sections into a new image
#             pixel_section = sums[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
#                             int(x * mag_ratio):int(x * mag_ratio + mag_ratio),
#                             z]
#             binned_image[y, x, z] = np.sum(pixel_section) # Take the sum value of the section and bin it to the camera.
#     print(z)



# plt.imshow(scaling_array[:,:,0])
# plt.show()
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



# x = np.load("1offset.npy")
# # print(np.max(x))
# # sam.savetiffs("output2.tif", x)
# array= np.zeros((100,100))
#
# for i in range(0, x.shape[2]-1):
#     array[i%100, i//100] = np.sum(x[:,:,i])
#     print(i//100, i%100)
# #
# plt.imshow(array)
# plt.imshow(x[:,:,5550])
# plt.show()





### INPUTS ###
#  Laser PSF #
# xy_size = 101
# pixel_size = 0.005
# stack_size = 40
# #
# #
# # ### PSF Generation ###
# # # Made a 3D PSF
# # # Each pixel = 5 nanometres.
# # # Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
# laserPSF = sam.radial_PSF(xy_size, pixel_size, stack_size)
# laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.
# #
# laserPSF = laserPSF / laserPSF.sum()          # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)
# #
# #
# # ### SAMPLE PARAMETERS ###
# # # Made a point in centre of 2D array
# point = np.zeros((xy_size, xy_size, laserPSF.shape[2]))
# # # point[50, 50, laserPSF.shape[2] // 2 + 8] = 1
# # # point[100, 100, laserPSF.shape[2] // 2 + 4] = 1
# point[00:laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1
# #
# # a = sam.array_multiply(point, laserPSF, 99, 99)
# #
# # # scan = scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))
# # #
# # #
# # # for i in range(0, point.shape[2]):
# # #     scan[:,:,i] = signal.fftconvolve(laserPSF[:, :, i], a[:, :, i], mode="same")
# # #     scan[:,:,i] = np.rot90(sam.kernel_filter_2D(laserPSF[:, :, i], a[:, :, i]), 2)        # When running a larger image over a smaller one it rotates the resulting info.
# # #
# # # Multiply by pinhole
# # summer = np.sum(scan,2)
# # circle_pinhole = sam.circle_mask(scan, 20, (125, 125))  # Produces a simple circle mask centred at x,y.
# # pinhole_sum = summer * circle_pinhole
# # #
# plt.subplot(121)
# plt.imshow(np.sum(point, 2))
# plt.imshow
# plt.subplot(122)
# plt.imshow(array)
# plt.show()


# plt.imshow(a[:,:,50])
# plt.show()


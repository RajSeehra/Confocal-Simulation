from PIL import Image
import numpy as np
import os
import microscPSF as msPSF
from scipy import signal
from fractions import Fraction
import pandas as pd
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, regionprops_table


def Gaussian_Map(image_size, offset, centre_x, centre_y, width, amplitude):
    # Image Generation
    x, y = np.meshgrid(np.linspace(-image_size[1]//2, image_size[1]//2, image_size[1]),
                       np.linspace(-image_size[0]//2, image_size[0]//2, image_size[0]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2)
    intensity = offset + amplitude * np.exp(-(dist ** 2 / (2.0 * width ** 2)))
    return intensity


### SAVE/LOADING FUNCTION ###
def get_file_list(dir):
    # dir = '/ugproj/Raj/Flash4/'
    file_list = []
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            file_name = dir + '/' + file
            file_list.append(file_name)
    return file_list


# single frame
def savetiff(file_name, data):
    images = Image.fromarray(data[:, :])
    images.save(file_name)


# single frame
def image_open(file):
    # open the file
    file_name = file
    img = Image.open(file_name)
    # generate the array and apply the image data to it.
    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    imgArray[:, :, 0] = img
    img.close()


# multi stack tiffs
def loadtiffs(file_name):
    img = Image.open(file_name)
    #print('The Image is', img.size, 'Pixels.')
    #print('With', img.n_frames, 'frames.')

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.float32)
    for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
    img.close()
    return (imgArray)


# as above
def savetiffs(file_name, data):
    images = []
    for I in range(np.shape(data)[2]):
        images.append(Image.fromarray(data[:, :, I]))
        images[0].save(file_name, save_all=True, append_images=images[1:])

### SAMPLE/PSF FUNCTIONS ###
def emptysphere3D(array, radius, sphere_centre):
    """ A function that takes an array size, sphere centre and radius and constructs an empty sphere.

    Parameters
    ----------
    array : arraylike
        the size of the array you wish for the sphere to be held in.
    radius : int
        the radius of the sphere. uniform in x, y, and z.
    sphere_centre : tuple
        the centre coordinates of the sphere.

    Returns
    ----------
    ones : ndarray
        an array of the same size as the input with ones denoting the sphere position values.

    """
    # PARAMETERS FOR CIRCLE MASK
    centre_x = sphere_centre[0]
    centre_y = sphere_centre[1]
    centre_z = sphere_centre[2]
    a_x = -centre_x                         # distance of left edge of screen relative to centre x.
    b_x = array.shape[1] - centre_x         # distance of right edge of screen relative to centre x.
    a_y = -centre_y                         # distance of top edge of screen relative to centre y.
    b_y = array.shape[0] - centre_y         # distance of bottom edge of screen relative to centre y.
    a_z = - centre_z                        # distance of z edge of screen relative to centre z.
    b_z = array.shape[2] - centre_z         # distance of z edge of screen relative to centre x.

    r = radius
    # Produce circle mask, ones grid = to original file and cut out.
    y, x, z = np.ogrid[a_y:b_y, a_x:b_x, a_z:b_z]       # produces a list which collapses to 0 at the centre in x and y
    mask1 = x*x + y*y + z*z <= r*r                      # produces a true/false array where the centre is true.
    mask2 = x*x + y*y + z*z - radius*3 <= r*r           # produces a second mask which helps construct our edge.
    ones= np.zeros((array.shape[1], array.shape[0], array.shape[2]))
    # combine the two mask to produce a sphere and then cut out the centre.
    ones[mask2] = 1                     # uses the mask to turn the zeroes to ones in the TRUE zone of mask.
    ones[mask1] = 0                     # uses the mask to turn the ones to zeroes in the TRUE zone of mask.

    return ones


def radial_PSF(xy_size, pixel_size=5, stack_size=40, wavelength = 0.600):
    # Radial PSF
    mp = msPSF.m_params  # Microscope Parameters as defined in microscPSF. Dictionary format.

    pixel_size = pixel_size  # In microns... (step size in the x-y plane)
    xy_size = xy_size  # In pixels.

    z_depth = (stack_size * pixel_size)/2

    pv = np.arange(-z_depth, z_depth, pixel_size)  # Creates a 1D array stepping up by denoted pixel size,
    # Essentially stepping in Z.

    psf_xy1 = msPSF.gLXYZFocalScan(mp, pixel_size, xy_size, pv, wvl=wavelength)  # Matrix ordered (Z,Y,X)

    psf_total = psf_xy1

    return psf_total

### KERNEL FILTERS FOR SLOW CONVOLUTION ###
# Can take 2D/3D arrays and kernels and convolute them.
def kernel_filter(data, matrix):
    image = data
    kernel = np.asarray(matrix)

    # Error check in case the matrix has an even number of sides.
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        print("The matrix has an even number of rows and/or columns. Please make them odd and run again.")

    if kernel.sum() != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print("This matrix's summation value is not equal to 1. This can change the final image.")
        print("Hence, the program has divided the matrix by the sum total to return it to a value of 1.")
        print(("This total value is: " + str(kernel.sum())))
        kernel = kernel / kernel.sum()
        # print(kernel)

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    # to determine if the file has multiple frames or not, runs a 2D kernel along a 3D array.
    if data.ndim == 3 and kernel.ndim == 2:
        print("2D kernel along a 3D array")
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h), (0, 0)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0], bordered_image.shape[2]))

        # Iterates the z, x and y positions.
        for z in range(0, bordered_image.shape[2]):
            for x in range(edge_cover_h, bordered_image.shape[1] - edge_cover_h):
                for y in range(edge_cover_v, bordered_image.shape[0] - edge_cover_v):
                    kernel_region = bordered_image[y - edge_cover_v:y + edge_cover_v + 1,
                                    x - edge_cover_h:x + edge_cover_h + 1, z]
                    k = (kernel * kernel_region).sum()
                    processed_image[y, x, z] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0] - edge_cover_v,
                          edge_cover_h:processed_image.shape[1] - edge_cover_h, :]

    elif data.ndim ==3 and kernel.ndim == 3:     # Runs a 3D matrix across a 3D array
        print("A 3D kernel across a 3D array")
        if kernel.shape[2] % 2 == 0:
            print("The matrix has an even number for Z depth. Please make them odd and run again.")
        edge_cover_d = (kernel.shape[2] - 1) // 2
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h), (edge_cover_d, edge_cover_d)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0], bordered_image.shape[2]))

        # Iterates the z, x and y positions.
        for z in range(edge_cover_d, bordered_image.shape[2] - edge_cover_d):
            for x in range(edge_cover_h, bordered_image.shape[1] - edge_cover_h):
                for y in range(edge_cover_v, bordered_image.shape[0] - edge_cover_v):
                    kernel_region = bordered_image[y - edge_cover_v:y + edge_cover_v + 1,
                                    x - edge_cover_h:x + edge_cover_h + 1, z - edge_cover_d:z + edge_cover_d + 1]
                    k = (kernel * kernel_region).sum()
                    processed_image[y, x, z] = k

        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0] - edge_cover_v,
                          edge_cover_h:processed_image.shape[1] - edge_cover_h, edge_cover_d:processed_image.shape[2]-edge_cover_d]

    else:
        print("A 2D kernel across a 2D array")
        # adds an edge to allow pixels at the border to be filtered too.
        bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h)))
        # Our blank canvas below.
        processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0]))

        # Iterates the x and y positions.
        for x in range(edge_cover_h, bordered_image.shape[1]-edge_cover_h):
            for y in range(edge_cover_v, bordered_image.shape[0]-edge_cover_v):
                kernel_region = bordered_image[y-edge_cover_v:y+edge_cover_v+1, x-edge_cover_h:x+edge_cover_h+1]
                k = (kernel * kernel_region).sum()
                processed_image[y, x] = k
        # Cuts out the image to be akin to the original image size.
        processed_image = processed_image[edge_cover_v:processed_image.shape[0]-edge_cover_v, edge_cover_h:processed_image.shape[1]-edge_cover_h]
        print(processed_image)
    return processed_image

def kernel_filter_2D(data, matrix):
    image = data
    kernel = np.asarray(matrix)

    if kernel.sum() == 0:
        print('Empty kernel')
    elif kernel.sum() != 1:       # Quick check to ensure the kernel matrix is within parameters.
        print(("This total value is: " + str(kernel.sum())))
        # kernel = kernel / kernel.sum()

    # Takes the filter size and allows for a rectangular matrix.
    edge_cover_v = (kernel.shape[0] - 1) // 2
    edge_cover_h = (kernel.shape[1] - 1) // 2

    print("A 2D kernel across a 2D array")
    # adds an edge to allow pixels at the border to be filtered too.
    bordered_image = np.pad(image, ((edge_cover_v, edge_cover_v), (edge_cover_h, edge_cover_h)))
    # Our blank canvas below.
    processed_image = np.zeros((bordered_image.shape[1], bordered_image.shape[0]))

    # Iterates the x and y positions.
    for x in range(edge_cover_h, bordered_image.shape[1] - edge_cover_h):
        for y in range(edge_cover_v, bordered_image.shape[0] - edge_cover_v):
            kernel_region = bordered_image[y - edge_cover_v:y + edge_cover_v + 1, x - edge_cover_h:x + edge_cover_h + 1]
            k = (kernel * kernel_region).sum()
            processed_image[y, x] = k
    # Cuts out the image to be akin to the original image size.
    processed_image = processed_image[edge_cover_v:processed_image.shape[0] - edge_cover_v,
                      edge_cover_h:processed_image.shape[1] - edge_cover_h]

    return processed_image


### PINHOLE MASK ###
def circle_mask(array, radius, centre_xy):
    # PARAMETERS FOR CIRCLE MASK
    centre_x = centre_xy[0]
    centre_y = centre_xy[1]
    a_x = -centre_x                         # distance of left edge of screen relative to centre x.
    b_x = array.shape[1] - centre_x         # distance of right edge of screen relative to centre x.
    a_y = -centre_y                         # distance of top edge of screen relative to centre y.
    b_y = array.shape[0] - centre_y         # distance of bottom edge of screen relative to centre y.

    r = radius
    # Produce circle mask, ones grid = to original file and cut out.
    y, x = np.ogrid[a_y:b_y, a_x:b_x]                 # produces a list which collapses to 0 at the centre in x and y
    mask = x*x + y*y <= r*r                           # produces a true/false array where the centre is true.
    ones= np.zeros((array.shape[1], array.shape[0]))
    ones[mask] = 1                                    # uses the mask to turn the zeroes to 1 in the TRUE zone of mask.
    # ones = ones / np.sum(ones)              # Normalised, unnecessary?
    return ones

##### MAIN BODY FUNCTIONS #####
def array_multiply(base_array, offset_array, x_pos, y_pos):
    # Using a cropping system this function multiplies two arrays at a certain position in space.
    offset_array_centre_dist_x = (offset_array.shape[1]) // 2
    offset_array_centre_dist_y = (offset_array.shape[0]) // 2


    # Base array pad values
    ba_left_pad = -(x_pos - offset_array_centre_dist_x) + 1
    if ba_left_pad < 0:
        ba_left_pad = 0
    ba_right_pad = x_pos + offset_array_centre_dist_x - base_array.shape[1]
    if x_pos + offset_array_centre_dist_x < base_array.shape[1]:
        ba_right_pad = 0
    ba_top_pad = -(y_pos - offset_array_centre_dist_y) + 1
    if ba_top_pad < 0:
        ba_top_pad = 0
    ba_bottom_pad = y_pos + offset_array_centre_dist_y - base_array.shape[0]
    if y_pos + offset_array_centre_dist_y < base_array.shape[0]:
        ba_bottom_pad = 0

    # Produce the bordered base image.
    ba_brd_image = np.pad(base_array, ((ba_top_pad, ba_bottom_pad), (ba_left_pad, ba_right_pad), (0,0)),"minimum")

    # Offset image pad values
    oi_left_pad = x_pos - offset_array_centre_dist_x - 1
    if oi_left_pad < 0:
        oi_left_pad = 0
    oi_right_pad = ba_brd_image.shape[1] - oi_left_pad - offset_array.shape[1]
    if oi_right_pad < 0:
        oi_right_pad = 0
    oi_top_pad = y_pos - offset_array_centre_dist_y - 1
    if  oi_top_pad < 0:
        oi_top_pad = 0
    oi_bottom_pad = ba_brd_image.shape[0] - oi_top_pad - offset_array.shape[1]
    if oi_bottom_pad < 0:
        oi_bottom_pad = 0

    # Produce the bordered offset image
    oi_brd_image = np.pad(offset_array, ((oi_top_pad, oi_bottom_pad), (oi_left_pad, oi_right_pad), (0, 0)), 'minimum')

    ### Checks for the values of the pads.
    # print(ba_left_pad,ba_right_pad,ba_top_pad,ba_bottom_pad)
    # print(oi_left_pad,oi_right_pad,oi_top_pad,oi_bottom_pad)

    # Now to finally multiply the two same sized arrays...
    multiplied_array = oi_brd_image * ba_brd_image

    # Extract the original section.
    multiplied_array = multiplied_array[ba_top_pad:ba_top_pad+base_array.shape[0],
                                        ba_left_pad:ba_left_pad+base_array.shape[1],
                                        :]

    return multiplied_array


def stage_scanning(laserPSF, point):
    # Produce an array that will receive the data we collect.
    laser_illum = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))  # Laser x sample
    scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))         # Laser illum conv w/ psf
    sums = np.zeros((point.shape[1], point.shape[0], int(point.shape[1] * point.shape[0])))  # z sum of scan.

    # Counter to track our z position/frame.
    counter = 0

    # Iterates across the array to produce arrays illuminated by the sample, with a laser blurred by the first lens.
    for x in range(0, point.shape[1]):
        for y in range(0, point.shape[0]):
            # Multiplies the PSF multiplied laser with the sample. The centre of the sample is moved to position x,y
            # on the laser array, as this is a stage scanning microscope.
            laser_illum = array_multiply(laserPSF, point, x, y)
            # Convolute the produced array with the PSF to simulate the second lens.
            for i in range(0, point.shape[2]):
                scan[:,:,i] = np.rot90(signal.fftconvolve(laserPSF[:,:,i], laser_illum[:,:,i], mode="same"),2)
                # scan[:,:,i] = np.rot90(sam.kernel_filter_2D(laserPSF[:, :, i], laser_illum[:, :, i]), 2)        # When running a larger image over a smaller one it rotates the resulting info.
            print("x:", x, " and y:", y)
            # Flatten and sum z stack.
            z_sum = np.sum(scan, 2)

            # Add to the collection array
            sums[:,:,counter] = z_sum
            print("Counter: ", counter)
            counter = counter + 1

    return sums


# Takes the scanned data set (sums) and the magnification ratio and upscales the value to produce 2 numbers and an array
# array_upscale = the value (in each direction) to which the array needs to be upscaled to allow integer based binning.
# upscale_ground_to_camera_num = the value which we will use to determine the number of pixel from scanned will be later
#                                used per each bin. e.g. if this were 15, then 15 of the upscaled pixels in x and 15 in
#                                y will be used to form the square to sum up and be placed into the binned array.
# upscaling_array = returns an array which is multiplied in x and y by the array_upscale. Pixels from the original array
#                   are distributed to a subselection of pixels and to maintain the overall value, we assume this
#                   distribution is equal for all pixels.
#                   The array is of shape(sums.shape[x]*array_upscale, sums.shape[y]*array_upscale, z)
def upscale(scanned_data, mag_ratio):
    # Turn the magfrac into a rational fraction
    mag_ratio = Fraction(mag_ratio).limit_denominator()
    # The upscale value for the array.
    array_upscale = mag_ratio.denominator
    # The upscale value for the number of pixels to read back.
    upscale_ground_to_camera_num = mag_ratio.numerator

    upscaling_array = np.zeros((scanned_data.shape[0] * array_upscale, scanned_data.shape[1] * array_upscale, scanned_data.shape[2]))
    for z in range(0, scanned_data.shape[2]):
        for y in range(0, scanned_data.shape[0]):
            for x in range(0, scanned_data.shape[1]):
                upscaling_array[y * array_upscale:y * array_upscale + array_upscale,
                                x * array_upscale:x * array_upscale + array_upscale,
                                z] \
                                 = scanned_data[y, x, z] / (array_upscale ** 2)
                print(x, y, z)
    return upscaling_array, upscale_ground_to_camera_num


def binning(sums, bin_array, mag_ratio):
    """ Forms a binned image from the scanned image and magnification ratio onto an empty bin_array.

    Parameters
    ----------
    sums:
        The summed value array to be binned.
    bin_array:
        An empty bin to which the function adds the data to be binned to.
    mag_ratio:
        the magnification ratio, usually a division of the ground truth pixel size and the camera pixel size.
        Further divided by the magnification.

    """
    for z in range(0, bin_array.shape[2]):
        for y in range(0, bin_array.shape[0]):
            for x in range(0, bin_array.shape[1]):
                # Takes the convoluted and summed data and bins the sections into a new image
                pixel_section = sums[int(y * mag_ratio):int(y * mag_ratio + mag_ratio),
                                     int(x * mag_ratio):int(x * mag_ratio + mag_ratio),
                                     z]
                bin_array[y, x, z] = np.sum(pixel_section)  # Take the sum value of the section and bin it to the camera.
        print(z)

    return bin_array

### NOISE FUNCTION ###

def read_noise(data, read_mean=2, read_std=2):
    # Build read noise. Always positive so we take the absolute values.
    read_noise = np.random.normal(read_mean, read_std / np.sqrt(2), np.shape(data))
    read_noise= np.abs(read_noise)
    return read_noise


def shot_noise(sqrt_mean_signal, data):
    shot_noise = np.random.poisson(sqrt_mean_signal, (np.shape(data)))
    return shot_noise




### ISM ###
def pixel_cutter(array, x_position, y_position, window_size_x=10, window_size_y=10, frame=0):
    # Single frame at a time... Will adapt and fix.
    imgArray = array[:, :, frame]
    # frame = frame - 1
    x = window_size_x
    y = window_size_y

    # imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    # imgArray[:, :, frame] = img
    # img.close()

    # Assign centre and ascertain coords for image, the -1 is to ensure even spreading either side of the centre point.
    xcoordmin = x_position - int(x / 2)
    xcoordmax = x_position + int(x / 2)+1
    ycoordmin = y_position - int(y / 2)
    ycoordmax = y_position + int(y / 2)+1

    # check no negative numbers
    if xcoordmin < 0:
        xcoordmin = 0
    if xcoordmax > imgArray.shape[0]:
        xcoordmax = imgArray.shape[0]
    if ycoordmin < 0:
        ycoordmin = 0
    if ycoordmax > imgArray.shape[1]:
        ycoordmax = imgArray.shape[1]
    # Plotting the area.
    print(imgArray[int(ycoordmin):int(ycoordmax), int(xcoordmin):int(xcoordmax)])
    return imgArray[int(ycoordmin):int(ycoordmax), int(xcoordmin):int(xcoordmax)]


# Example
# x = pixel_cutter("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Images/bacteria.tif", 15,100, 15,15, 0)
# print(np.shape(x))
# plt.imshow(x)
# plt.show()


def centre_collection(dataset):
    """ Takes an input 2D or 3D array and returns the coordinates of the brightest point for each stack.

    Parameters
    ----------
     dataset:
        The array to extract the centre values from, it will do this individually for each array and works on 2D and 3D.

    Return
    ----------
     coordinates: arraylike
        An array containing the coordinate values in order (y,x,z). It is of size (3, dataset.shape[2])
    """
    # Datachecks to make sure we are being passed an appropriate array.
    if dataset.ndim == 3:
        z_data = dataset.shape[2]
    elif dataset.ndim == 2:
        dataset = np.expand_dims(dataset, 2)
        z_data = dataset.shape[2]
    else:
        raise ValueError("This array is not in 2D or 3D space... It contains an array of", dataset.ndim, "dimensions.")

    # make an empty array to extract the coordinates to.
    coordinates = np.zeros((3, z_data))

    # Load the images iteratively in z.
    for z in range(0, z_data):
        # Pulls our the index of the array and assign them in order y,x to the arraylist.
        coordinates[0, z], coordinates[1, z] = np.unravel_index(np.argmax(dataset[:, :, z]), (dataset.shape[0], dataset.shape[1]))
        # Assign the z position for future reference
        coordinates[2, z] = z
    return coordinates


def gaussian_weighting(dataset):

    # for each frame of the dataset we aim to multiply by a gaussian centred on the coordinate value given.
    for z in range(0, dataset.shape[2]):
        centre_x = dataset.shape[1]//2
        centre_y = dataset.shape[0]//2

        dataset[:,:,z] = dataset[:,:,z] * (Gaussian_Map(dataset.shape, 0, centre_x, centre_y, 1, 1)/np.max(Gaussian_Map(dataset.shape, 0, centre_x, centre_y, 1, 1)))

    return dataset

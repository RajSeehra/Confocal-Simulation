from PIL import Image
import numpy as np
import Confocal_Processing as proc
import microscPSF as msPSF
from scipy import signal
from fractions import Fraction
from skimage.transform import resize


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
    ones= np.zeros((array.shape[0], array.shape[1], array.shape[2]))
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


def stage_scanning(laserPSF, point, emission_PSF):
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

            # Add Shot Noise to the laser Illumination.
            mean_signal = np.mean(laser_illum)
            s_noise = shot_noise(np.sqrt(laser_illum), laser_illum)
            print("Shot noise generated.")
            laser_illum = laser_illum + s_noise
            print("Shot noise added.")

            # Convolute the produced array with the PSF to simulate the second lens.
            for i in range(0, point.shape[2]):
                scan[:,:,i] = np.rot90(signal.fftconvolve(emission_PSF[:,:,i], laser_illum[:,:,i], mode="same"),2)
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


### NOISE FUNCTIONS ###
def read_noise(data, read_mean=2, read_std=2):
    # Build read noise. Always positive so we take the absolute values.
    read_noise = np.random.normal(read_mean, read_std / np.sqrt(2), np.shape(data))
    read_noise= np.abs(read_noise)
    return read_noise


def shot_noise(sqrt_mean_signal, data):
    shot_noise = np.random.poisson(sqrt_mean_signal, (np.shape(data)))
    return shot_noise

### CONFOCAL ###
def confocal(pinhole_sum, point):
    # Set up the confocal array
    conf_array = np.zeros((point.shape[0], point.shape[1]))

    # Iterate through the z stack and sum the values and add them to the appropriate place on the image.
    for i in range(0, pinhole_sum.shape[2]):
        conf_array[i % point.shape[0], i // point.shape[1]] = np.sum(pinhole_sum[:, :, i])
        print(i // point.shape[1], i % point.shape[0])

    return conf_array

### ISM ###
def ISM(pinhole_sum, point, pinhole_radius, scale=16):
    # Finds the centre of the image and collects the position data to then cut out a zone around the centre at the size
    # of the pinhole.
    y = proc.centre_collection(pinhole_sum)

    Cut_out_pinhole = 2 * pinhole_radius
    cut_section = np.zeros((pinhole_radius*2, pinhole_radius*2, pinhole_sum.shape[2]))
    for i in range(0, pinhole_sum.shape[2]):
        cut_section[:, :, i] = proc.pixel_cutter(pinhole_sum, y[1, i], y[0, i], Cut_out_pinhole, Cut_out_pinhole, i)
        print("cutting:", i)

    scale = scale
    # The final image is initially at an arbitrary value greater than the original image.
    final_image = np.zeros((point.shape[0] * scale, point.shape[1] * scale))

    # Each frame from cut_section is upscaled to half said arbitrary size and then added to the array at the appropriate
    # position.
    upscaled_PSFs = np.zeros(
        (cut_section.shape[0] * (scale // 2), cut_section.shape[1] * (scale // 2), cut_section.shape[2]))
    for z in range(0, cut_section.shape[2] - 1):
        upscaled_PSFs[:, :, z] = resize(cut_section[:, :, z], (upscaled_PSFs.shape[0], upscaled_PSFs.shape[1]),
                                        mode="constant", cval=0)

    # Pad the final image so we can add the upscaled PSFs to it.
    final_image = np.pad(final_image, ((upscaled_PSFs.shape[0] // 2, upscaled_PSFs.shape[0] // 2),
                                       (upscaled_PSFs.shape[1] // 2, upscaled_PSFs.shape[1] // 2)))

    # Place the arrays in the final image array at the appropriate position pos*16.
    for i in range(0, pinhole_sum.shape[2]):
        x = i // point.shape[1]
        y = i % point.shape[0]
        final_image[y * scale:y * scale + upscaled_PSFs.shape[0], x * scale:x * scale + upscaled_PSFs.shape[1]] = \
            final_image[y * scale:y * scale + upscaled_PSFs.shape[0],
            x * scale:x * scale + upscaled_PSFs.shape[1]] + upscaled_PSFs[:, :, i]
        print("placing:", i // point.shape[1], i % point.shape[0])

    final_image = final_image[upscaled_PSFs.shape[0] // 2:-upscaled_PSFs.shape[0] // 2,
                  upscaled_PSFs.shape[1] // 2:-upscaled_PSFs.shape[1] // 2]

    # Need downscaling that is less than the upscaling and then array where each 'pixel' = the downscaling area. 2x base.

    downscale = np.zeros((point.shape[0] * 2, point.shape[1] * 2))
    for y in range(0, downscale.shape[0]):
        for x in range(0, downscale.shape[1]):
            # Takes the convoluted and summed data and bins the sections into a new image
            pixel_section = final_image[int(y * scale // 2):int(y * scale // 2 + scale // 2),
                            int(x * scale // 2):int(x * scale // 2 + scale // 2)]
            downscale[y, x] = np.sum(pixel_section)  # Take the sum value of the section and bin it to the camera.
        print("downscale:", y)

    return downscale
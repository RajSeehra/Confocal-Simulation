from PIL import Image
import numpy as np
import os
import microscPSF as msPSF


def pixel_cutter(file_name, x_position, y_position, window_size_x=10, window_size_y=10, frame=0):
    img = Image.open(file_name)
    frame = frame - 1
    x = window_size_x
    y = window_size_y

    imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
    imgArray[:, :, frame] = img
    img.close()

    # Assign centre and ascertain coords for image, the -1 is to ensure even spreading either side of the centre point.
    xcoordmin = x_position - int(x / 2)-1
    xcoordmax = x_position + int(x / 2)
    ycoordmin = y_position - int(y / 2)-1
    ycoordmax = y_position + int(y / 2)

    # check no negative numbers
    if xcoordmin < 0:
        xcoordmin = 0
    if xcoordmax > img.size[0]:
        xcoordmax = img.size[0]
    if ycoordmin < 0:
        ycoordmin = 0
    if ycoordmax > img.size[1]:
        ycoordmax = img.size[1]

    # Plotting the area.
    return imgArray[ycoordmin:ycoordmax, xcoordmin:xcoordmax, frame]


# Example
# x = pixel_cutter("/Users/RajSeehra/University/Masters/Semester 2/Teaching_python-master/Images/bacteria.tif", 15,100, 15,15, 0)
# print(np.shape(x))
# plt.imshow(x)
# plt.show()


def Gaussian_Map(image_size, offset, centre_x, centre_y, width, amplitude):
    # Image Generation
    x, y = np.meshgrid(np.linspace(-image_size[1]//2, image_size[1]//2, image_size[1]),
                       np.linspace(-image_size[0]//2, image_size[0]//2, image_size[0]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2)
    intensity = offset + amplitude * np.exp(-(dist ** 2 / (2.0 * width ** 2)))
    return intensity


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


def radial_PSF(xy_size, pixel_size=5, stack_size=40):
    # Radial PSF
    mp = msPSF.m_params  # Microscope Parameters as defined in microscPSF. Dictionary format.

    pixel_size = pixel_size  # In microns... (step size in the x-y plane)
    xy_size = xy_size  # In pixels.

    z_depth = (stack_size * pixel_size)/2

    pv = np.arange(-z_depth, z_depth, pixel_size)  # Creates a 1D array stepping up by denoted pixel size,
    # Essentially stepping in Z.

    psf_xy1 = msPSF.gLXYZParticleScan(mp, pixel_size, xy_size, pv)  # Matrix ordered (Z,Y,X)

    psf_total = psf_xy1

    return psf_total


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
    y, x = np.ogrid[a_x:b_x, a_y:b_y]                 # produces a list which collapses to 0 at the centre in x and y
    mask = x*x + y*y <= r*r                           # produces a true/false array where the centre is true.
    ones= np.zeros((array.shape[1], array.shape[0]))
    ones[mask] = 1                                    # uses the mask to turn the zeroes to 1 in the TRUE zone of mask.
    ones = ones / np.sum(ones)              # Normalised
    return ones


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
    ba_brd_image = np.pad(base_array, ((ba_top_pad, ba_bottom_pad), (ba_left_pad, ba_right_pad), (0,0)))

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
    oi_brd_image = np.pad(offset_array, ((oi_top_pad, oi_bottom_pad), (oi_left_pad, oi_right_pad), (0, 0)))

    print(ba_left_pad,ba_right_pad,ba_top_pad,ba_bottom_pad)
    print(oi_left_pad,oi_right_pad,oi_top_pad,oi_bottom_pad)

    # Now to finally multiply the two same sized arrays...
    multiplied_array = oi_brd_image * ba_brd_image

    # Extract the original section.
    multiplied_array = multiplied_array[ba_top_pad:ba_top_pad+base_array.shape[0],
                                        ba_left_pad:ba_left_pad+base_array.shape[1],
                                        :]

    return multiplied_array

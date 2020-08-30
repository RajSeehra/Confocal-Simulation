import numpy as np
import os
from PIL import Image


### OPEN AND SAVE FUNCTIONS ###
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


### BASIC FUNCTIONS ###
def Gaussian_Map(image_size, offset, centre_x, centre_y, width, amplitude):
    # Image Generation
    x, y = np.meshgrid(np.linspace(-image_size[1]//2, image_size[1]//2, image_size[1]),
                       np.linspace(-image_size[0]//2, image_size[0]//2, image_size[0]))
    dist = np.sqrt((x-centre_x) ** 2 + (y+centre_y) ** 2)
    intensity = offset + amplitude * np.exp(-(dist ** 2 / (2.0 * width ** 2)))
    return intensity


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

    # Assign centre and ascertain coords for image, the +0.5 is to ensure even spreading either side of the centre point
    xcoordmin = x_position - int(x // 2)
    xcoordmax = x_position + int((x / 2)+0.5)
    ycoordmin = y_position - int(y // 2)
    ycoordmax = y_position + int((y / 2)+0.5)

    # Setup empty pad values. This is to account for edges and corners as we wish to return an array of appropriate size
    l_pad = 0
    r_pad = 0
    t_pad = 0
    b_pad = 0


    # check no negative numbers
    if xcoordmin < 0:
        l_pad = -xcoordmin
        xcoordmin = 0
    if xcoordmax > imgArray.shape[1]:
        r_pad = xcoordmax - imgArray.shape[1]
        xcoordmax = imgArray.shape[1]
    if ycoordmin < 0:
        t_pad = -ycoordmin
        ycoordmin = 0
    if ycoordmax > imgArray.shape[0]:
        b_pad = ycoordmax - imgArray.shape[0]
        ycoordmax = imgArray.shape[0]

    # print(l_pad,r_pad,t_pad,b_pad)
    # Plotting the area.
    array = imgArray[int(ycoordmin):int(ycoordmax), int(xcoordmin):int(xcoordmax)]

    if l_pad > 0 or t_pad > 0 or r_pad > 0 or b_pad > 0:
        array = np.pad(array, ((int(t_pad), int(b_pad)), (int(l_pad), int(r_pad))))

    return array


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

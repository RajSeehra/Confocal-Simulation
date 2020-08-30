import PySimpleGUI as sg
import microscPSF as msPSF
import GUI_Conf_main as conf
import

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


sg.theme('DarkBlack')   # Adds a theme
# All the stuff inside the window. With parameters to define the user input within certain limits.
layout = [  [sg.Text('Imaging Mode', size=(30, 1)),sg.Spin(("Widefield", "Confocal", "ISM"), initial_value="Widefield", size=(11, 1))],
            [sg.Text('XY Size (pixels)', size=(30, 1)), sg.InputText("5", size=(6, 30), justification="right")],
            [sg.Text('Stack Size (pixels)', size=(30, 1)), sg.InputText("5", size=(6, 30), justification="right")],

            [sg.Text('Ground Truth Pixel Size (nm)', size=(30, 1)), sg.InputText("5", size=(6,30), justification="right")],
            [sg.Text('Laser Power', size=(30, 1)), sg.Slider((0,20), 8, 2, 1, 'h', size=(30,20))],
            [sg.Text('Exposure Time (s)', size=(30, 1)), sg.InputText("1",size=(6,30), justification="right")],

            [sg.Text('Excitation Wavelength (250-700)', size=(30, 1)), sg.Slider((200,700), 440, 10, 100, 'h')],
            [sg.Text('Emission Wavelength (250-700)', size=(30, 1)), sg.Slider((200,700), 440, 10, 100, 'h')],
            [sg.Text('Numerical Aperture', size=(30, 1)), sg.Slider((0, 2), 1.4, 0.1, 0.5, 'h')],

            [sg.Text('Camera Pixel Size (nm)', size=(30, 1)), sg.InputText("6500", size=(6,30), justification="right")],
            [sg.Text('Magnification', size=(30, 1)), sg.Spin(("1", "10", "20", "40", "100", "200"), initial_value="100", size=(5,1))],

            [sg.Text('Quantum Efficiency (0-1)', size=(30, 1)), sg.Slider((0,1), 0.75, 0.1, 0.25, 'h')],
            [sg.Text('Gain', size=(30, 1)), sg.InputText("2", size=(6,30), justification="right")],

            [sg.Text('Read Noise Mean', size=(30, 1)), sg.InputText("2", size=(6,30), justification="right")],
            [sg.Text('Read Noise Standard Deviation', size=(30, 1)), sg.InputText("2", size=(6,30), justification="right")],

            [sg.Text('Include Shot Noise?', size=(30,1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5,1))],

            [sg.Text('Fixed Pattern Deviation', size=(30, 1)), sg.InputText("0.001", size=(6,30), justification="right")],

            [sg.Text('Preview?', size=(30, 1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5, 1))],
            [sg.Text('Save?', size=(30, 1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5, 1))],
            [sg.Text('Filename? (TIFF by default)', size=(30, 1)), sg.InputText("Camera_image")],

            [sg.Text('Progress Bar', size=(30, 1)), sg.ProgressBar(100, bar_color=("blue", "white"), key="Progress Bar")],
            [sg.Button('Run'), sg.Button('Cancel')] ]

# Creates the Window based on the layout.
window = sg.Window('Camera Emulator', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()   # Reads the values entered into a list.
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        break

    # Define the progress bar to allow updates
    progress_bar = window.FindElement('Progress Bar')

    ### INPUTS ###
    #  Laser, PSF and Sample #
    pixel_size = 0.02      # Ground truth pixel size in microns
    xy_size = 100           # xy size for both laser and sample.
    stack_size = 100         # Z depth for the PSF
    laser_power = 10 ** float(values[4])       # Laser power per second in should be microwatts where 1 count = 1 microwatt. average in live cell is 15000 microwatt
    exposure_time = 1       # seconds of exposure

    # PSF
    excitation_wavelength = 0.480    # Wavelength in microns
    emission_wavelength = 0.540      # Wavelength in microns
    NA = 1.4                # Numerical aperture
    ### Dont touch the below line ###
    msPSF.m_params["NA"] = NA   # alters the value of the microscope parameters in microscPSF. Has a default value of 1.4

    # PINHOLE #
    pinhole_radius = 100        # Radius of the pinhole in pixels.
    offset = 0                # Offsets the pinhole. Meant to help increase resolution but needs to be implemented in fourier reweighting..
    # CAMERA
    camera_pixel_size = 6   # Camera pixel size in microns. usual sizes = 6 microns or 11 microns
    magnification = 100     # Lens magnification
    msPSF.m_params["M"] = magnification   # alters the value of the microscope parameters in microscPSF. Has a default value of 100
    QE = 0.7                # Quantum Efficiency
    gain = 2                # Camera gain. Usually 2 per incidence photon
    count = 100             # Camera artificial count increase.
    # NOISE
    include_read_noise = "Y" # Y/N. Include read noise
    read_mean = 2           # Read noise mean level
    read_std = 2             # Read noise standard deviation level
    include_shot_noise = "Y" # Y/N. Include shot noise
    fix_seed = "Y"        # Y/N to fix the Shot noise seed.
    include_fixed_pattern_noise = "Y" # Y/N. Include fixed pattern noise
    fixed_pattern_deviation = 0.001  # Fixed pattern standard deviation. usually affects 0.1% of pixels.
    # MODE #
    mode = "Widefield"       # Mode refers to whether we are doing "Widefield NEED TO ADD", Confocal or ISM imaging.
    # SAVE
    Preview = "Y"
    SAVE = "N"              # Save parameter, input Y to save, other parameters will not save.
    filename = "Conf_pinhole100_200x200"   # filename without file format as a string, saves as tiff


    print(laser_power)


#
# def stage_scanning(laserPSF, point, emission_PSF):
#     # Produce an array that will receive the data we collect.
#     laser_illum = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))  # Laser x sample
#     scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))         # Laser illum conv w/ psf
#     sums = np.zeros((point.shape[1], point.shape[0], int(point.shape[1] * point.shape[0])))  # z sum of scan.
#     # Counter to track our z position/frame.
#     counter = 0
#
#     # Iterates across the array to produce arrays illuminated by the sample, with a laser blurred by the first lens.
#     for x in range(0, point.shape[1]):
#         for y in range(0, point.shape[0]):
#             # Multiplies the PSF multiplied laser with the sample. The centre of the sample is moved to position x,y
#             # on the laser array, as this is a stage scanning microscope.
#             laser_illum = conf.array_multiply(laserPSF, point, x, y)
#
#             # Add Shot Noise to the laser Illumination.
#             mean_signal = np.mean(laser_illum)
#             s_noise = conf.shot_noise(np.sqrt(laser_illum), laser_illum)
#             print("Shot noise generated.")
#             laser_illum = laser_illum + s_noise
#             print("Shot noise added.")
#
#             # Convolute the produced array with the PSF to simulate the second lens.
#             for i in range(0, point.shape[2]):
#                 scan[:,:,i] = np.rot90(signal.fftconvolve(emission_PSF[:,:,i], laser_illum[:,:,i], mode="same"),2)
#                 # scan[:,:,i] = np.rot90(sam.kernel_filter_2D(laserPSF[:, :, i], laser_illum[:, :, i]), 2)        # When running a larger image over a smaller one it rotates the resulting info.
#             print("x:", x, " and y:", y)
#             # Flatten and sum z stack.
#             z_sum = np.sum(scan, 2)
#
#             # Add to the collection array
#             sums[:,:,counter] = z_sum
#             print("Counter: ", counter)
#             counter = counter + 1
#
#     return sums
#
# def radial_PSF(xy_size, xy_pixel_size=5, z_pixel_size = 10, stack_size=40, wavelength = 0.600):
#     """ A function that produces a PSF for a given wavelength in an array size given to it.
#
#     Parameters
#     ----------
#     xy_size : int
#         The number of pixels in the array in xy. e.g. 100x100 = 100.
#     pixel_size: float
#
#     stack_size:
#     wavelength:
#
#     Returns
#     ---------
#
#
#     """
#
#     # Radial PSF
#     mp = msPSF.m_params  # Microscope Parameters as defined in microscPSF. Dictionary format.
#
#     xy_pixel_size = xy_pixel_size  # In microns... (step size in the x-y plane)
#     xy_size = xy_size  # In pixels.
#
#     z_depth = (stack_size * z_pixel_size)/2
#
#     pv = np.arange(-z_depth, z_depth, z_pixel_size)  # Creates a 1D array stepping up by denoted pixel size,
#     # Essentially stepping in Z.
#
#     psf_xy1 = msPSF.gLXYZFocalScan(mp, xy_pixel_size, xy_size, pv, wvl=wavelength)  # Matrix ordered (Z,Y,X)
#
#     psf_total = psf_xy1
#
#     return psf_total





# xy_size = 100
# z = 100
# Made a point in centre of 2D array
# point = np.zeros((xy_size, xy_size, z))
# point[25, 25, 1] = intensity
# point[75, 75, -1] = intensity
# point[xy_size//2, xy_size//2, z // 2] = 1
# point[laserPSF.shape[0]//2, laserPSF.shape[1]//2+20, laserPSF.shape[2] // 2] = intensity

## Spherical ground truth  ##
# radius = 40
# sphere_centre = (point.shape[0]//2, point.shape[1]//2, point.shape[2] // 2)
# point = conf.emptysphere3D(point, radius, sphere_centre)
## More Complex Spherical Ground Truth ##
# sample = point
# sample = conf.emptysphere3D(sample, int(sample.shape[0]*0.4), (sample.shape[1]//2, sample.shape[0]//2, sample.shape[2]//2))
# sample2 = conf.emptysphere3D(sample, int(sample.shape[0]*0.25), (sample.shape[1]//2.5, sample.shape[0]//2.5, sample.shape[2]//2.5))
# sample3 = conf.emptysphere3D(sample, int(sample.shape[0]*0.05), (sample.shape[1]//1.4, sample.shape[0]//1.4, sample.shape[2]//1.7))
# sample4 = conf.emptysphere3D(sample, int(sample.shape[0]*0.05), (sample.shape[1]//2.5, sample.shape[0]//1.4, sample.shape[2]//1.4))
# point = (sample+sample2+sample3+sample4)





# def convolve(emission, laser):
#     scany = np.rot90(signal.fftconvolve(emission,laser, mode='same'),2)
#     return scany
#
#
# def doit():
#     items = [(emission_PSF[:,:,i],laser_illum[:,:,i]) for i in range(emission_PSF.shape[2])]
#     pool = mp.Pool(mp.cpu_count())
#     results = pool.starmap(convolve, items)
#     pool.close()
#
#     return results
#

# if __name__ == '__main__':
#     xy = 100
#     z = 100
#
#     laserPSF = radial_PSF(xy, 0.02,0.02, z, 0.480)
#     laserPSF = np.moveaxis(laserPSF, 0, -1)  # The 1st axis was the z-values. Now in order y,x,z.
#
#     laserPSF = (laserPSF / laserPSF.sum()) * (10000 * 1)
#
#     emission_PSF = radial_PSF(xy, 0.02,0.02, z, 0.540)
#     emission_PSF = np.moveaxis(emission_PSF, 0, -1)  # The 1st axis was the z-values. Now in order y,x,z.
#     emission_PSF = (emission_PSF / emission_PSF.sum())
#
#     sample = np.zeros((xy, xy, z))
#
#     sample[laserPSF.shape[0]//2, laserPSF.shape[1]//2, laserPSF.shape[2] // 2] = 1
#     point = sample
#
#     # sample = conf.emptysphere3D(sample, int(sample.shape[0] * 0.45),
#     #                             (sample.shape[1] // 2, sample.shape[0] // 2, sample.shape[2] // 2))
#     # sample2 = conf.emptysphere3D(sample, int(sample.shape[0] * 0.25),
#     #                              (sample.shape[1] // 2.5, sample.shape[0] // 2.5, sample.shape[2] // 2.5))
#     # sample3 = conf.emptysphere3D(sample, int(sample.shape[0] * 0.05),
#     #                              (sample.shape[1] // 1.4, sample.shape[0] // 1.4, sample.shape[2] // 1.7))
#     # sample4 = conf.emptysphere3D(sample, int(sample.shape[0] * 0.05),
#     #                              (sample.shape[1] // 2.5, sample.shape[0] // 1.4, sample.shape[2] // 1.4))
#     #
#     # point = sample + sample2 + sample3 + sample4
#
#     # EDIT
#
#     scan = np.zeros((laserPSF.shape[1], laserPSF.shape[0], laserPSF.shape[2]))  # Laser illum conv w/ psf
#     sums = np.zeros((point.shape[1], point.shape[0], int(point.shape[1] * point.shape[0])))  # z sum of scan.
#     # Counter to track our z position/frame.
#     counter = 0
#
#     x = point.shape[1]//2 +10
#     y = point.shape[0]//2 +10
#
#     laser_illum = conf.array_multiply(point, laserPSF, x, y)
#
#     # Add Shot Noise to the laser Illumination.
#     # s_noise = conf.shot_noise(np.sqrt(laser_illum), laser_illum)
#     # print("Shot noise generated.")
#     # laser_illum = laser_illum + s_noise
#     print("Shot noise added.")
#
#     # Old
#     start = time.time()
#     for i in range(0, point.shape[2]):
#         scan[:, :, i] = np.rot90(signal.fftconvolve(emission_PSF[:, :, i], laser_illum[:, :, i], mode="same"),2)
#     end = time.time()
#     print("For Loop:",end-start)
#     results = scan
#     z_sum = np.sum(results, 2)
#
#     # New
#     start = time.time()
#     results2 = doit()
#     end = time.time()
#     print("Multiprocessing:",end - start)
#     z_sum_2 = np.sum(results2, 0)
#
#
#     items = [(emission_PSF[:,:,i],laser_illum[:,:,i]) for i in range(emission_PSF.shape[2])]
#     # print(np.sum(emission_PSF[:,:,0]), np.sum(laser_illum[:,:,0]))
#     a,b = items[0]
#     c = emission_PSF[:,:,0] - a
#     d = laser_illum[:,:,0] - b
#     c1 = np.sum(c)
#     d1 = np.sum(d)
#
#     # 0.033296035829118414 good
#     # 0.22621940709885408 bad
#
#     # plt.subplot(121)
#     plt.imshow(z_sum)
#     # plt.subplot(122)
#     # plt.imshow(z_sum_2)
#     plt.show()




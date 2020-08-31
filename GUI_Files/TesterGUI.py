import PySimpleGUI as sg
import microscPSF as msPSF
import numpy as np
from GUI_Files import GUI_Confocal_Processing as proc
from GUI_Files import GUI_Confocal_Main_Func as conf
from GUI_Files import GUI_Data_Check as dc


def int_checker(variable, min, max):
    # if last character in input element is invalid, remove it
    global data_check, event, values, window
    check = True

    if event == variable and values[variable] and values[variable][-1] not in ('0123456789') or values[variable] == '':
        window[variable].update(values[variable][:-1])
        check = False
        data_check = data_check +1

    for i in range(0, len(values[variable])):
        if values[variable][i] not in ('0123456789'):
            window[variable].update(values[variable][:i-1])
            check = False
            data_check = data_check + 1

    if check == True:
        if event == variable and (int(values[variable]) < int(min) or int(values[variable]) > max):
            window[variable].update(values[variable][:-1])
            data_check = data_check +1


def float_checker(variable, min, max):
    # if last character in input element is invalid, remove it
    global data_check, event, values, window
    check = True

    if event == variable and values[variable] and values[variable][-1] not in ('0123456789.') or values[variable] == '' or values[variable]== '.':
        window[variable].update(values[variable][:-1])
        check = False
        data_check = data_check +1

    for i in range(0, len(values[variable])):
        if values[variable][i] not in ('0123456789.'):
            window[variable].update(values[variable][:i-1])
            check = False
            data_check = data_check + 1


    if check == True:
        if event == variable and (float(values[variable]) < float(min) or float(values[variable]) > float(max)):
            window[variable].update(values[variable][:-1])
            data_check = data_check +1


sg.theme('DarkBlack')   # Adds a theme
sg.set_options(font=("arial",(14)),text_justification="right")
# All the stuff inside the window. With parameters to define the user input within certain limits.
layout = [  [sg.Text(text='Confocal Simulation Program', size=(60,1), pad=((25,5),(0,5)),
                     border_width=4, background_color="blue", justification="centre", font=("arial", 26, "bold"))],
            [sg.Text("")],
            [sg.Text('Imaging Mode', size=(25, 1 )),sg.Spin(("Widefield", "Confocal", "ISM"), key="IM", initial_value="Widefield", size=(11, 1)),
            sg.Text("Ground Truth Sample", size=(25,1)), sg.Spin(("Point Sample", "Empty Sphere", "Complex Sphere"), key="GTS", initial_value="Point Sample", size=(14,1))],

            [sg.Text(text='\n Laser and Sample:', size=(22,2), pad=((25,5),(0,5)),
                     border_width=4,text_color="red", justification="left", font=("arial", 16, "bold"))],
            [sg.Text('XY Size (pixels)', size=(25, 1)), sg.InputText("100", size=(6, 30), key="XY", enable_events=True),
            sg.Text('Stack Size (pixels)', size=(25, 1)), sg.InputText("100", size=(6, 30), key="SS", enable_events=True),
            sg.Text('Ground Truth Pixel Size (nm)', size=(25, 1)), sg.InputText("20", size=(6,30), key="PS", enable_events=True)],
            [sg.Text('Laser Power', size=(25, 1)), sg.Slider((0,20), 8, 2, 1, 'h', size=(30,20), key="LP"),
             sg.Text('Exposure Time (s)', size=(25, 1)), sg.InputText("1", size=(6, 30), key="ET", enable_events=True)],

            [sg.Text(text='PSF:', size=(10, 1), pad=((25, 5), (5, 5)),
                     border_width=4, text_color="red", justification="left", font=("arial", 16, "bold"))],
            [sg.Text('Excitation Wavelength (250-700)', size=(27, 1)), sg.Slider((200, 700), 440, 10, 100, 'h', key="EXW"),
            sg.Text('Numerical Aperture', size=(25, 1)), sg.Slider((0, 2), 1.4, 0.1, 0.5, 'h', key="NA")],[
            sg.Text('Emission Wavelength (250-700)', size=(27, 1)), sg.Slider((200, 700), 440, 10, 100, 'h', key="EMW")],

            [sg.Text(text='Pinhole:', size=(8, 1), pad=((25, 0), (5, 5)),
                     border_width=4, text_color="red", justification="left", font=("arial", 16, "bold")),
            sg.Text('Pinhole Radius', size=(14, 1)), sg.InputText("1", size=(6, 30), justification="right", key="PR", enable_events=True),
            sg.Text('Offset (Disabled)', size=(25, 1)), sg.InputText("0", size=(6, 30), justification="right", disabled=True, key="O", enable_events=True)],

            [sg.Text(text='\n Camera and Magnification:', size=(25, 2), pad=((25, 5), (5, 5)),
                     border_width=4, text_color="red", justification="left", font=("arial", 16, "bold"))],
            [sg.Text('Camera Pixel Size (nm)', size=(25, 1)), sg.InputText("6000", size=(6,30), justification="right", key="CPS", enable_events=True),
            sg.Text('Magnification', size=(25, 1)), sg.Spin(("1", "10", "20", "40", "100", "200"), initial_value="100", size=(5,1), key="M"),
            sg.Text('Gain', size=(25, 1)), sg.InputText("2", size=(6,30), justification="right", key="G", enable_events=True)],
            [sg.Text('Count', size=(25, 1)), sg.InputText("100", size=(6, 30), justification="right", key="C", enable_events=True),
            sg.Text('Quantum Efficiency (0-1)', size=(25, 1)), sg.Slider((0, 1), 0.75, 0.1, 0.25, 'h', key="QE")],

            [sg.Text("")],

            [sg.Text(text='Noise:', size=(6, 1), pad=((25, 0), (5, 5)), border_width=4, text_color="red", justification="left", font=("arial", 16, "bold")),

            sg.Text('Read Noise Mean', size=(15, 1)), sg.InputText("2", size=(7,30), justification="right", key="rm", enable_events=True),
             sg.Text('Read Noise Standard Deviation', size=(25, 1)), sg.InputText("2", size=(7,30), justification="right", key="rs", enable_events=True),
             sg.Text("Include Read Noise?", pad=((20,5),(5,5))), sg.CBox("", key="irn", default=True)],

            [sg.Text('Fix Shot Noise Seed?', size=(25, 1)), sg.CBox("", default=True, key="fs"),
             sg.Text("Include Shot Noise?", pad=((20,5),(5,5))), sg.CBox("", key="isn", default=True)],

            [sg.Text('Fixed Pattern Deviation', size=(25, 1)), sg.InputText("0.001", size=(7,30), justification="right", key="fpn", enable_events=True),
             sg.Text("Include Fixed Pattern Deviation?", pad=((20,5),(5,5))), sg.CBox("", key="ifpn", default=True)],

            [sg.Text(text='Output:', size=(10, 2), pad=((25, 5), (5, 5)),
                     border_width=4, text_color="red", justification="left", font=("arial", 16, "bold")),
            sg.Text('Preview?', size=(9, 1)), sg.Spin(("Y", "N"), initial_value="Y", size=(5, 1), key="P")],
            [sg.Text('Save?', size=(25, 1)), sg.Spin(("Y", "N"), initial_value="N", size=(5, 1), key="S"),
            sg.Text('Filename? (TIFF by default)', size=(25, 1)), sg.InputText("Camera_image", key="F", enable_events=True)],

            [sg.Text('Progress Bar', size=(25, 1)),
             sg.ProgressBar(100, size=(50,10), bar_color=("blue", "white"), key="Progress Bar"),
             sg.Text('Step', size=(20,1), key="Step"), sg.Text("0/0", key="Progression")],

            [sg.Button('Run'), sg.Button('Cancel')] ]

# Creates the Window based on the layout.
window = sg.Window('Camera Emulator', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()   # Reads the values entered into a list.
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        window.close()
        break

    # Define the progress bar to allow updates
    progress_bar = window.FindElement('Progress Bar')
    step = window.FindElement('Step')
    progression = window.FindElement('Progression')

    step.Update("Checking Data")
    ### DATA CHECKING ###
    data_check = 0
    int_checker("PS", 1, 1000)
    int_checker("XY", 1, np.inf)
    int_checker("SS", 1, np.inf)
    float_checker("ET", 0, 100)
    int_checker("CPS", 1, 30000)
    int_checker("G", 1, 500)
    int_checker("C", 0, 1000)
    float_checker("rm", 0, 200)
    float_checker("rs", 0, 200)
    float_checker("fpn", 0, 1)
    if data_check == 0:
        float_checker("PR", 0, int(values["XY"]))
        int_checker("O", 0, int(values["XY"]))

    steps = 0
    if values["IM"] == "Widefield":
        steps = 15
    if values["IM"] == "Confocal":
        steps = 15
    if values["IM"] == "ISM":
        steps = 15

    if data_check == 0 and event in 'Run':      # Only runs if data is checked and within parameters.
        step.Update("Inputting data")
        progression("1/" + str(steps))

        ### INPUTS ###
        #  Laser, PSF and Sample #
        sample_type = values["GTS"]
        pixel_size = int(values["PS"])/1000      # Ground truth pixel size in microns
        xy_size = int(values["XY"])           # xy size for both laser and sample.
        stack_size = int(values["SS"])         # Z depth for the PSF
        laser_power = 10 ** float(values["LP"])       # Laser power per second in should be microwatts where 1 count = 1 microwatt. average in live cell is 15000 microwatt
        exposure_time = float(values["ET"])       # seconds of exposure

        progress_bar.UpdateBar((6/29 * 100))

        # PSF
        excitation_wavelength = float(values["EXW"])    # Wavelength in microns
        emission_wavelength = float(values["EMW"])      # Wavelength in microns
        NA = float(values["NA"])                       # Numerical aperture
        ### Dont touch the below line ###
        msPSF.m_params["NA"] = NA   # alters the value of the microscope parameters in microscPSF. Has a default value of 1.4

        progress_bar.UpdateBar((9/29 * 100))

        # PINHOLE #
        pinhole_radius = float(values["PR"])        # Radius of the pinhole in pixels.
        offset = int(values["O"])                # Offsets the pinhole. Meant to help increase resolution but needs to be implemented in fourier reweighting..
        # CAMERA
        camera_pixel_size = int(values["CPS"])/1000   # Camera pixel size in microns. usual sizes = 6 microns or 11 microns
        magnification = int(values["M"])     # Lens magnification
        msPSF.m_params["M"] = magnification   # alters the value of the microscope parameters in microscPSF. Has a default value of 100
        QE = float(values["QE"])                # Quantum Efficiency
        gain = int(values["G"])                # Camera gain. Usually 2 per incidence photon
        count = int(values["C"])             # Camera artificial count increase.

        progress_bar.UpdateBar((16/29 * 100))

        # NOISE
        include_read_noise = values["irn"]         # Y/N. Include read noise
        read_mean = float(values["rm"])           # Read noise mean level
        read_std = float(values["rs"])             # Read noise standard deviation level
        include_shot_noise = values["isn"]         # Y/N. Include shot noise
        fix_seed = values["fs"]                   # Y/N to fix the Shot noise seed.
        include_fixed_pattern_noise = values["ifpn"]   # Y/N. Include fixed pattern noise
        fixed_pattern_deviation = float(values["fpn"])  # Fixed pattern standard deviation. usually affects 0.1% of pixels.

        progress_bar.UpdateBar((23/29 * 100))

        # MODE #
        mode = values["IM"]       # Mode refers to whether we are doing "Widefield NEED TO ADD", Confocal or ISM imaging.
        # SAVE
        Preview = values["P"]
        SAVE = values["S"]              # Save parameter, input Y to save, other parameters will not save.
        filename = values["F"]   # filename without file format as a string, saves as tiff

        progress_bar.UpdateBar((29/29 * 100))

        print(pixel_size, xy_size)


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




import numpy as np
from GUI_Files import GUI_Confocal_Main_Func as confmain
from GUI_Files import GUI_Confocal_Processing as proc
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import microscPSF as msPSF
from fractions import Fraction
from scipy import signal

# This program takes a 3D sample and simulates a stage scanning microscope. Hence the sample is 'moved'
# relative to the detector.
# The final output is a 3D image whereby the sample has passed through: a lens as denoted by a convolution with a
# radial PSF, and a pinhole simulated by a layer wise convolution with a gaussian spacial filter.
# Array sizes need to be odd to work with the convolution as a centre point needs to be determinable.

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

            [sg.Text('Step', size=(20,1), key="Step"), sg.Text("0/0", key="Progression")],

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
    step = window.FindElement('Step')
    progression = window.FindElement('Progression')

    if values["IM"] == "Widefield":
        sg.set_options()

    step.Update("Checking Data")
    ### DATA CHECKING ###
    data_check = 0
    int_checker("PS", 1, 1000)  # Ground Truth Pixel Size
    int_checker("XY", 1, 2000)  # XY image size
    int_checker("SS", 1, 2000)  # Z stack size
    float_checker("ET", 0, 100)  # Exposure Time
    int_checker("CPS", 1, 30000)  # Camera Pixel Size
    int_checker("G", 1, 500)    # Gain
    int_checker("C", 0, 1000)   # Count
    float_checker("rm", 0, 200)  # read noise mean
    float_checker("rs", 0, 200)  # read noise standard deviation
    float_checker("fpn", 0, 1)  # Fixed Pattern Noise
    if data_check == 0:
        float_checker("PR", 0, int(values["XY"]))
        int_checker("O", 0, int(values["XY"]))

    steps = 0
    if values["IM"] == "Widefield":
        steps = 8
    if values["IM"] == "Confocal":
        steps = 10
    if values["IM"] == "ISM":
        steps = 10


    if data_check == 0 and event in 'Run':      # Only runs if data is checked and within parameters.
        step.Update("Inputting data")
        progression.Update("1/" + str(steps))

        ### INPUTS ###
        #  Laser, PSF and Sample #
        sample_type = values["GTS"]
        pixel_size = int(values["PS"])/1000      # Ground truth pixel size in microns
        xy_size = int(values["XY"])           # xy size for both laser and sample.
        stack_size = int(values["SS"])         # Z depth for the PSF
        laser_power = 10 ** float(values["LP"])       # Laser power per second in should be microwatts where 1 count = 1 microwatt. average in live cell is 15000 microwatt
        exposure_time = float(values["ET"])       # seconds of exposure

        # PSF
        excitation_wavelength = float(values["EXW"])/1000    # Wavelength in microns
        emission_wavelength = float(values["EMW"])/1000      # Wavelength in microns
        NA = float(values["NA"])                       # Numerical aperture
        ### Dont touch the below line ###
        msPSF.m_params["NA"] = NA   # alters the value of the microscope parameters in microscPSF. Has a default value of 1.4

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

        # NOISE
        include_read_noise = values["irn"]         # Y/N. Include read noise
        read_mean = float(values["rm"])           # Read noise mean level
        read_std = float(values["rs"])             # Read noise standard deviation level
        include_shot_noise = values["isn"]         # Y/N. Include shot noise
        fix_seed = values["fs"]                   # Y/N to fix the Shot noise seed.
        include_fixed_pattern_noise = values["ifpn"]   # Y/N. Include fixed pattern noise
        fixed_pattern_deviation = float(values["fpn"])  # Fixed pattern standard deviation. usually affects 0.1% of pixels.

        # MODE #
        mode = values["IM"]       # Mode refers to whether we are doing "Widefield NEED TO ADD", Confocal or ISM imaging.
        # SAVE
        Preview = values["P"]
        SAVE = values["S"]              # Save parameter, input Y to save, other parameters will not save.
        filename = values["F"]   # filename without file format as a string, saves as tiff


        ###### MAIN PROGRAM ######
        # Order of the program: produce sample and PSF, multiply, convolute and sum them in z for each xy point,
        #                       bin the images as though they have been magnified to the cameras pixels, then
        #                       process the data through confocal or ISM methods to produce the final image.

        ### PSF Generation ###
        # Made a 3D PSF
        # Each pixel = x nanometres.
        # Greater xy size the more PSF is contained in the array. 255 seems optimal, but 101 works and is much faster.
        step.Update("Generating Laser")
        progression.Update("2/" + str(steps))

        laserPSF = confmain.radial_PSF(xy_size, pixel_size, stack_size, excitation_wavelength)
        laserPSF = np.moveaxis(laserPSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.

        laserPSF = (laserPSF / laserPSF.sum()) * (laser_power * exposure_time)  # Equating to 1. (to do: 1 count =  1 microwatt,
                                                                                                # hence conversion to photons.)


        ### SAMPLE PARAMETERS ###
        step.Update("Generating Sample")
        progression.Update("3/"+ str(steps))

        if mode == "Widefield":
            intensity = 100000
        else:
            intensity = 1
        point = np.zeros((xy_size, xy_size, laserPSF.shape[2]))
        point = confmain.sample_selector(sample_type, point, intensity)


        ### STAGE SCANNING SO SAMPLE IS RUN ACROSS THE PSF (OR 'LENS') ###
        step.Update("Stage Scanning")
        progression.Update("4/" + str(steps))

        # Establish the PSF of the emission wwavelength as the light passes through the lens after scanning
        emission_PSF = confmain.radial_PSF(xy_size, pixel_size, stack_size, emission_wavelength)
        emission_PSF = np.moveaxis(emission_PSF, 0, -1)     # The 1st axis was the z-values. Now in order y,x,z.
        emission_PSF = (emission_PSF / emission_PSF.sum())  # Equating to 1. (to do: 1 count =  1 microwatt, hence conversion to photons.)

        # Takes the psf and sample arrays as inputs and scans across them.
        # Returns an array at the same xy size with a z depth of x*y.
        if mode == "Widefield":
            sums = signal.fftconvolve(point, emission_PSF, "same")
            for x in range(0, sums.shape[1]):
                for y in range(0,sums.shape[0]):
                    for z in range(0, sums.shape[2]):
                        if sums[y,x,z] < 0:
                            sums[y,x,z] = 0
            if include_shot_noise:
                shot_noise = confmain.shot_noise(np.sqrt(sums), sums, fix_seed)
                print("Shot noise added")
            else:
                shot_noise = 0
                print("Shot noise not added")
            sums = sums + shot_noise
        else:
            sums = confmain.stage_scanning(laserPSF, point, emission_PSF, include_shot_noise, fix_seed)


        ### CAMERA SETUP ###
        step.Update("Setting Up Camera")
        progression.Update("5/" + str(steps))
        # Camera sensor, based on optical magnification and pixel size.
        camerapixel_per_groundpixel = camera_pixel_size / pixel_size

        # Used to determine the number of the ground pixels that exist within each bin
        # We use the mag_ratio to step across the array and select out regions that are multiples of it out.
        mag_ratio = camerapixel_per_groundpixel / magnification
        print("Overall Image Binning (ground pixels per bin):", mag_ratio, "by", mag_ratio)


        ##### IMAGING TIME #####

        ### UPSCALING ###
        # If the mag_ratio is not a whole number then we have an issue with distributing the values to the new array accurately.
        # The below functions aims to upscale each 2D scan to allow for an integer based binning method.

        upscale = 0         # a simple counter to determine if upscaling has been done or not.
        upscaled_sum = 0
        upscale_mag_ratio = 0
        if Fraction(mag_ratio-int(mag_ratio)) != 0:
            step.Update("Upscaling")
            progression.Update("6.5/" + str(steps))

            upscaled_sum, upscale_mag_ratio = confmain.upscale(sums, mag_ratio)
            upscale = 1


        ### BINNING ###
        step.Update("Binning")
        progression.Update("6/" + str(steps))

        # Initialise an empty array, with a size calculated by the above ratios.
        # Gives us a rounded down number of pixels to bin into to prevent binning half a bin volume into an end pixel.
        # Bins each image collected at the detector to the camera pixel size.
        binned_image = np.zeros((int(sums.shape[0] // mag_ratio),
                                 int(sums.shape[1] // mag_ratio),
                                 int(sums.shape[2])))
        print("Binning Data to Camera")

        # If upscaling has occured we need to use different variables for the function.
        sums_list = [sums, upscaled_sum]
        mag_ratio_list = [mag_ratio, upscale_mag_ratio]

        # Actual binning step.
        binned_image = confmain.binning(sums_list[upscale], binned_image, mag_ratio_list[upscale])
        print("Data Binned")


        ### QUANTUM EFFICIENCY ###
        QE_image = binned_image * QE
        print("QE step")


        ### NOISE ###
        step.Update("Noise")
        progression.Update("7/" + str(steps))

        print("Creating noise.")
        if include_read_noise == "Y":
            read_noise = confmain.read_noise(QE_image, read_mean, read_std)
            print("Read noise generated.")
        else:
            read_noise = 0
        # Fix the seed for fixed pattern noise
        if include_fixed_pattern_noise == "Y":
            np.random.seed(100)
            fixed_pattern_noise = np.random.normal(1, fixed_pattern_deviation, (QE_image.shape[0],
                                                                                QE_image.shape[1],
                                                                                QE_image.shape[2]))
        else:
            fixed_pattern_noise = 1
        print("Fixed Pattern noise generated.")


        # Sum the noises and the image
        noisy_image = (QE_image + read_noise) * fixed_pattern_noise
        print("Combining noises with image. Complete!")


        ### GAIN, COUNT AND INTEGER ###
        step.Update("Gain and Count")
        progression.Update("8/" + str(steps))
        # Multiply by gain to convert from successful incidence photons and noise to electrons.
        print("All about them gains.")
        camera_gain = noisy_image * gain

        # 100 count added as this is what camera's do.
        print("Count on it.")
        camera_plusCount = camera_gain + count

        # Convert to integer as a camera output can only take integers
        # Conversion to: USER INT VALUE 16
        print("Integerising at the detector")
        camera_view = camera_plusCount.astype(np.uint16)


        ### PINHOLE ###
        # Multiply by pinhole, which is centered and offset.
        # Produces a simple circle mask centred at x,y.
        if mode == "Widefield":
            pinhole_sum = noisy_image
        else:
            step.Update("Pinhole")
            progression.Update("9/" + str(steps))

            circle_pinhole = proc.circle_mask(camera_view, pinhole_radius,
                                              (camera_view.shape[1] // 2 + offset, camera_view.shape[0] // 2 + offset))
            # Produce an empty array to place the results into.
            pinhole_sum = np.zeros((camera_view.shape[0], camera_view.shape[1], camera_view.shape[2]))

            # Multiplies the binned image with the circle pinhole at each z position.
            # This digitally adds our pinhole to the binned image at the size we desire.
            for i in range(0, camera_view.shape[2]):
                pinhole_sum[:, :, i] = camera_view[:, :, i] * circle_pinhole
            print("Pinholes added.")


        ### CONFOCAL SUMMING ###
        # Reconstruct the image based on the collected arrays.
        if mode == "Widefield":
            conf_array = np.sum(pinhole_sum,2)
            print("Widefield Image, Check")
        elif mode == "Confocal":
            step.Update("Constructing Confocal Image")
            progression.Update("10/" + str(steps))

            print("So it's CONFOCAL imaging time.")
            conf_array = confmain.confocal(pinhole_sum, point)
            print("CONFOCAL, DEPLOY IMAGE!!")

        elif mode == "ISM":
            step.Update("Constructing ISM Image")
            progression.Update("10/" + str(steps))

            print("ISM, a wise choice.")
            conf_array = confmain.ISM(pinhole_sum, point, pinhole_radius)
            print("ISM, I See More? Check the image to find out.")

        else:
            # Will become redundant once checks are in place.
            print("You didn't select an appropriate reconstruction form...")
            print("To save you some time we are outputting the Confocal based reconstruction of the image.")
            conf_array = confmain.confocal(pinhole_sum, point)


        ### SAVE, PREVIEW ###
        if Preview == "Y":
            plt.imshow(conf_array)
            plt.show()

        if SAVE == "Y":
            print("Saving Image")
            proc.savetiff(filename+".tif", conf_array)
            print("Image saved.")



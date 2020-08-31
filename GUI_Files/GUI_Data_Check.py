import PySimpleGUI
# Needs evaluation to allow greater data checking simultaneously.
def dict_datacheck_lesser_greater(data, lesser_values, greater_values):
    """ Will take a dictionary of variables and check if the check variable is in the ranges prescribed by the function.

    :param data:
        The dictionary containing the variables to be checked.
    :param lesser_values:
        The minimum value for the variable in question.
    :param greater_values:
        The maximum value for the variable.
    :return:
    """
    check_variable_names = [key for key in data.keys()]

    for i in range(0, len(check_variable_names)):
        check_variable_name = check_variable_names[i]
        test_variable = float(data[check_variable_name])

        lesser_value = float(lesser_values[i])
        greater_value = float(greater_values[i])

        while test_variable < lesser_value or test_variable > greater_value:
            data[check_variable_name] = input(check_variable_name + " is not in the range expected:"
                                              + str(lesser_value) + "-" + str(greater_value))
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




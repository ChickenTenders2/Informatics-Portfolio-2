##########################################################################################
# Task 1 [7 points out of 30] My first not-so-pretty image classifier
# Your first task is to implement the kNN classifier on your own. The template contains a range of functions
# you must implement and use appropriately for this task. The template also uses a range of functions implemented
# by the module leader to support you in this task, particularly relating to reading images and csv files accompanying
# this portfolio.
# You can start working on this task immediately. Please consult at the very least Week 2 materials.
##########################################################################################

import os
import Dummy
import Helper
import numpy
from typing import Callable
import Task_4

# Please replace with your student id, including the "c" at the beginning!!!
student_id = 'c23073897'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']

# Task 1 Objective (Personal Note): If k=3, the new photo looks most similar to these 3 training photos,
# and if 2 of these training photos are "Female", the new photo should be classified as "Female" too.

# Given a dictionary of classes and their occurrences, returns a class from the scheme that is most common in that
# dictionary. In case there are multiple candidates, it follows the order of classes in the scheme.
# The function returns empty string if the input dictionary is empty, does not contain any classes from the scheme,
# or if all classes in the scheme have occurrence of 0.
#
# INPUT: nearest_neighbours_classes
#                           : a dictionary that, for each and every class in the scheme, states the occurrence number
#                             of this class (this particular value is calculated elsewhere). You cannot assume
#                             any particular order of elements in the dictionary.
#
# OUTPUT: winner            : a class from classification scheme that is most common in the input variable.
#                             In case there are multiple candidates, it follows the order of classes in the scheme.
#                             Returns empty string if the input dictionary is empty, does not contain any classes
#                             from the scheme, or if all classes in the scheme have occurrence of 0
#

def getMostCommonClass(nearest_neighbours_classes: dict[str, int]) -> str:
    # Have fun! Below is just a dummy for returns, feel free to edit
    winner = ''

    # Check if dictionary is empty or doesn't contain classes from the scheme
    if not nearest_neighbours_classes or not any(cls in classification_scheme for cls in nearest_neighbours_classes):
        return ''
    
    # Find maximum occurrence count
    max_count = max(nearest_neighbours_classes.values()) if nearest_neighbours_classes else 0

    # If all classes have zero occurrences, return empty string
    if max_count == 0:
        return ''
    
    # Find classes with maximum count, following the order in the classification scheme 
    for cls in classification_scheme:
        if cls in nearest_neighbours_classes and nearest_neighbours_classes[cls] == max_count:
            winner = cls
            return winner

    # If no class is found, return empty string
    return ''

# The function finds the k nearest neighbours from measures_classes, and returns a dictionary made of classes
# and their occurrences based on these k nearest neighbours.
#
# INPUT:  measure_classes   : a list of tuples that contain two elements each - a distance/similarity value
#                             and class from scheme (in that order). You cannot assume that the input is
#                             pre-sorted in any way.
#         k                 : the value of k neighbours, greater than 0, not guaranteed to be smaller than data size.
#         similarity_flag   : a boolean value stating that the measure used to produce the values above is a distance
#                             (False) or a similarity (True)
# OUTPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#

def getClassesOfKNearestNeighbours(measures_classes: list[tuple[float, str]], k: int, similarity_flag: bool) -> (dict)[str, int]:
    # Have fun! Below is just a dummy for returns, feel free to edit

    # Initialize dictionary with all clases from the scheme set to 0 
    nearest_neighbours_classes = {cls: 0 for cls in classification_scheme}

    # Sort base on similarity or distance 
    # For similarity, higher values are better (descending order)
    # For distance, lower values are better (ascending order)
    sorted_measures = sorted(measures_classes, key=lambda x: x[0], reverse = similarity_flag)

    # Take most k elements (handle case where k is greater than available data)
    k_nearest = sorted_measures[:min(k, len(sorted_measures))]

    # Count occurrences of each class
    for _, cls in k_nearest:
        if cls in nearest_neighbours_classes:
            nearest_neighbours_classes[cls] += 1

    return nearest_neighbours_classes

# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this! You need to use all the other functions in the part of the template above.
#
# INPUT:  training_data       : a numpy array that was read from the training data csv
#         data_to_classify    : a numpy array  that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
#         k                   : the value of k neighbours, greater than 0, not guaranteed to be smaller than data size.
#         measure_func        : the function to be invoked to calculate similarity/distance (see Task 4 for
#                               some teacher-defined ones)
#         similarity_flag     : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#     most_common_class_func  : the function to be invoked to find the most common class among the neighbours
#                             (by default, it is the one from above)
# get_neighbour_classes_func  : the function to be invoked to find the classes of nearest neighbours
#                             (by default, it is the one from above)
#         read_func           : the function to be invoked to find to read and resize images
#                             (by default, it is the Helper function)
#  OUTPUT: classified_data    : a numpy array which expands the data_to_classify with the results on how your
#                             classifier has classified a given image.

def kNN(training_data: numpy.typing.NDArray, data_to_classify: numpy.typing.NDArray, k: int, measure_func: Callable,
        similarity_flag: bool, most_common_class_func=getMostCommonClass,
        get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=Helper.readAndResize) -> numpy.typing.NDArray:
    
    # This sets the header list
    classified_data = numpy.array([['Path', 'ActualClass', 'PredictedClass']])
    # Have fun!

    # Process each entry in data_to_classify (excluding header)
    for i in range(1, len(data_to_classify)):
        image_path = data_to_classify[i][0] # Path is first column
        actual_class = data_to_classify[i][1] # ActualClass is second column

        # Read the image to classify
        image_to_classify = read_func(image_path)

        # If image couldn't be read, skip classification
        if image_to_classify.size == 0:
            classified_data = numpy.append(classified_data, [[image_path, actual_class, "Error"]], axis=0)
            continue

        # Calculate similarity/distance to all training samples
        measures_classes = []
        for j in range(1, len(training_data)):
            train_image_path = training_data[j][0] # Path is first column
            train_class = training_data[j][1] # ActualClass is second column

            # Read training image
            train_image = read_func(train_image_path)

            # If training image couldn't be read, skip
            if train_image.size == 0:
                continue

            # Calculate similarity/distance
            measure = measure_func(image_to_classify, train_image)
            measures_classes.append((measure, train_class))

        # Get class distribution of k nearest neighbors
        neighbours_classes = get_neighbour_classes_func(measures_classes, k, similarity_flag)

        # Predict class by finding most common class among neighbors
        predicted_class = most_common_class_func(neighbours_classes)

        # Add result to classfied data
        classified_data = numpy.append(classified_data, [[image_path, actual_class, predicted_class]], axis=0)

    return classified_data

##########################################################################################
# Do not modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################

# This function reads the necessary arguments from input (see parse_arguments function in Helper file),
# and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.

def main():
    opts = Helper.parseArguments()
    if not opts:
        print("Missing input. Read the README file.")
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = Helper.readCSVFile(opts['training_data'])
    data_to_classify = Helper.readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    if training_data.size == 0 or data_to_classify.size == 0:
        print("Not all of the input data is present, cannot run kNN. Exiting Task 1.")
        return
    if opts['k'] is None or opts['k']<1:
        print("Value of k is missing from input or too small, cannot run kNN. Exiting Task 1.")
        return
    if opts['simflag'] is None:
        print("Similarity flag is missing from input, cannot run kNN. Exiting Task 1.")
        return
    print('Running kNN')
    try:
        result = kNN(training_data, data_to_classify, opts['k'], eval(opts['measure']), opts['simflag'])
    except NameError as nerror:
        print(nerror)
        print("Wrong measure function name was passed to the function, please double check the function name. "
              "For example, try 'Task_4.computePSNRSimilarity' and make sure you have not deleted any imports "
              "from the template.")
        return
    except TypeError as terror:
        print("Measure function is incorrect or missing, please double check the input. "
              "For example, try 'Task_4.computePSNRSimilarity' and make sure you have not deleted any imports "
              "from the template.")
        return
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        Helper.writeCSVFile(out, result)

if __name__ == '__main__':
    main()
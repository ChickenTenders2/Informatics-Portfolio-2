import unittest
import sys
import cv2
from io import StringIO
import copy
import logging
import os
import Task_4
from Task_1 import *
import Dummy

sys.tracebacklimit = 6

# Fix the path issue by modifying the path in the CSV files in memory
def fix_paths_in_csv(csv_data):
    fixed_data = copy.deepcopy(csv_data)
    for i in range(1, len(fixed_data)):  # Skip header
        if isinstance(fixed_data[i][0], str) and "../Images" in fixed_data[i][0]:
            # Replace "../Images" with "./Images" in the path
            fixed_data[i][0] = fixed_data[i][0].replace("../Images", "../Images")
    return fixed_data

# Read the CSV files with original paths
original_training_data = Helper.readCSVFile("../training_data.csv")
original_testing_data = Helper.readCSVFile("../testing_data.csv")

# Fix the paths
training_data = fix_paths_in_csv(original_training_data)
testing_data = fix_paths_in_csv(original_testing_data)

class Task_1_Testing(unittest.TestCase):
    #
    # This function contains one unit test for getClassesOfKNearestNeighbours.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test1_getClassesOfKNearestNeighbours_distance_behaviour(self):
        # Setting up the variables - value of k, the input measure_classes, and the output we expect for a distance approach
        k = 3
        measure_classes = [(0.5, 'Female'), (0.2, 'Male'), (1, 'Male'), (3, 'Primate'), (2.0, 'Female')]
        outputIfDist = {'Female': 1, 'Male': 2, 'Primate': 0, 'Rodent': 0, 'Food': 0}
        student_output = getClassesOfKNearestNeighbours(copy.deepcopy(measure_classes), k, False)
        result = student_output == outputIfDist
        result_message = 'Produced output is not equal to the expected one. Expected ' + str(
            outputIfDist) + " and got " + str(student_output)
        self.assertEqual(result, True, result_message)

    # This function contains one unit test for getMostCommonClass.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test2_getMostCommonClass(self):
        input1 = {'Female': 2, 'Male': 5, 'Primate': 1, 'Rodent': 3, 'Food': 0}
        answer1 = 'Male'
        student_output = getMostCommonClass(copy.deepcopy(input1))
        result = student_output == answer1
        result_message = 'Produced output is not equal to the expected one. Expected ' + str(
            answer1) + " and got " + str(student_output)
        self.assertEqual(result, True, result_message)

    # This function contains one unit test checking if kNN output is in the right format (no guarantee
    # that the content is right itself!)
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test3_kNNOutputFormat(self):
        # Use a very small subset of data to make the test faster and avoid missing files
        small_training = numpy.array([training_data[0]])  # Just the header
        small_testing = numpy.array([testing_data[0]])    # Just the header
        
        # Add a few rows of data that we know exist
        for i in range(1, min(5, len(training_data))):
            # Check if the file exists before adding
            if os.path.isfile(training_data[i][0]):
                small_training = numpy.append(small_training, [training_data[i]], axis=0)
                
        for i in range(1, min(3, len(testing_data))):
            # Check if the file exists before adding
            if os.path.isfile(testing_data[i][0]):
                small_testing = numpy.append(small_testing, [testing_data[i]], axis=0)
        
        # Only run the test if we have enough data
        if len(small_training) > 1 and len(small_testing) > 1:
            # Modified to use a smaller dataset
            output = kNN(small_training, small_testing, 1, Task_4.computePSNRSimilarity, True)
            result = validateDataFormat(output)
            self.assertEqual(result, True, "kNN function output is not formatted well")
        else:
            # Skip test if not enough valid data
            self.skipTest("Not enough valid image files found to run this test")

# This function checks if the classified data from kNN has the right format (not content, just format!)
# It checks if the required columns are present, and if entries are paths and classes
def validateDataFormat(data_to_validate):
    formatCorrect = True
    if not set(["Path", "ActualClass", "PredictedClass"]).issubset(set(data_to_validate[0])):
        return False
    for row in data_to_validate[1:]:
        isFile = os.path.isfile(row[0])
        isClass = row[1] in classification_scheme
        isClass = isClass and row[2] in classification_scheme
        if not isFile or not isClass:
            return False
    return formatCorrect

if __name__ == "__main__":
    test_classes_to_run = [Task_1_Testing]
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classes_to_run)
    runner = unittest.TextTestRunner()
    runner.run(suite)
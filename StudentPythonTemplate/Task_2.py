##########################################################################################
# Task 2 [6 points out of 30] Basic evaluation
# Evaluate your classifier. On your own, implement a method that will create a confusion matrix based on the provided
# classified data. Then implement methods that will return TPs, FPs and FNs based on the confusion matrix.
# From these, implement binary precision, recall and f-measure, and their macro counterparts.
# Finally, implement the multiclass version of accuracy.
# Remember to be mindful of edge cases (the approach for handling them is explained in lecture slides).
# The template contains a range of functions you must implement and use appropriately for this task.
# The template also uses a range of functions implemented by the module leader to support you in this task,
# particularly relating to reading images and csv files accompanying this portfolio.
# You can start working on this task immediately. Please consult at the very least Week 3 materials.
##########################################################################################

import Helper
import Dummy
import numpy
from Task_1 import classification_scheme

# This function computes the confusion matrix based on the provided data.
#
# INPUT: classified_data   : a numpy array containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description. Remember, this data contains
#                            header row!
# OUTPUT: confusion_matrix : a numpy array representing the confusion matrix computed based on the classified_data.
#                            The order of elements MUST be the same as in the classification scheme.
#                            The columns correspond to actual classes and rows to predicted classes.
#                            In other words, confusion_matrix[0] should be understood
#                            as the row of values predicted as Female, and [row[0] for row in confusion_matrix] as the
#                            column of values that were actually Female (independently of if the classified data
#                            contained Female entries or not).

def confusionMatrix(classified_data: numpy.typing.NDArray) -> numpy.typing.NDArray:
    # Have fun! Below is just a dummy for returns, feel free to edit
    # Initialize confusion matrix with zeros
    # Rows represent predicted classes
    # Columns represent actual classes
    n_classes = len(classification_scheme)
    confusion_matrix = numpy.zeros((n_classes, n_classes), dtype=int)

    # Skip the header row
    for i in range(1, len(classified_data)): 
        actual_class = classified_data[i][1] # ActualClass column
        predicted_class = classified_data[i][2] # PredictedClass column

        # Find indices for actual and predicted classes
        if actual_class in classification_scheme and predicted_class in classification_scheme:
            actual_idx = classification_scheme.index(actual_class)
            predicted_idx = classification_scheme.index(predicted_class)
            confusion_matrix[predicted_idx][actual_idx] += 1

    return confusion_matrix

# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# INPUT: confusion_matrix : the numpy array representing the confusion matrix computed based on the classified_data.
#                           The order of elements is the same as  in the classification scheme.
#                           The columns correspond to actual classes and rows to predicted classes.
# OUTPUT: a list of ints representing appropriate true positive, false positive or false
#         negative values per a given class, in the same order as in the classification scheme. For example, tps[1]
#         corresponds to TPs for Male class.

def computeTPs(confusion_matrix: numpy.typing.NDArray) -> list[int]:
    # Have fun! Below is just a dummy for returns, feel free to edit
    tps = [] # True Positives
    for i in range(len(confusion_matrix)):
        tps.append(confusion_matrix[i][i]) # Diagonal elements are TPs
    return tps

def computeFPs(confusion_matrix: numpy.typing.NDArray) -> list[int]:
    # Have fun! Below is just a dummy for returns, feel free to edit
    fps = [] # False Positives
    for i in range(len(confusion_matrix)):
        # Sum of row minus the TP
        fps.append(sum(confusion_matrix[i]) - confusion_matrix[i][i])
    return fps

def computeFNs(confusion_matrix: numpy.typing.NDArray) -> list[int]:
    # Have fun! Below is just a dummy for returns, feel free to edit
    fns = [] # False Negatives
    for i in range(len(confusion_matrix)):
        # Sum of column minus the TP
        fns.append(sum(confusion_matrix[:,i]) - confusion_matrix[i][i])
    return fns

# These functions compute the binary measures based on the provided values. Not all measures use all of the values.
#
# INPUT: tp, fp, fn : the values of true positives, false positive and negatives
# OUTPUT: appropriate evaluation measure created using the binary approach.

def computeBinaryPrecision(tp: int, fp: int, fn: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    precision = float(0)
    # Handle division by zero
    if tp + fp == 0:
        return precision
    precision = tp / (tp + fp)
    return precision

def computeBinaryRecall(tp: int, fp: int, fn: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    recall = float(0)
    # Handle division by zero
    if tp + fn == 0:
        return recall
    recall = tp / (tp + fn)
    return recall

def computeBinaryFMeasure(tp: int, fp: int, fn: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    f_measure = float(0)
    precision = computeBinaryPrecision(tp, fp, fn)
    recall = computeBinaryRecall(tp, fp, fn)

    # Handle division by zero
    if precision + recall == 0:
        return f_measure
    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure

# These functions compute the evaluation measures based on the provided values - macro precision, macro recall,
# macro f-measure, and accuracy (multiclass version). Not all measures use of all the values.
# You are expected to use appropriate binary counterparts when needed (binary recall for macro recall, binary precision
# for macro precision, binary f-measure for macro f-measure).
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and number of classified entries
#                         in the classified data (aka, don't count the header!)
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps: list[int], fps: list[int], fns: list[int], data_size: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    precision = float(0)
    
    # Calculate precision for each class and average
    precisions = []
    for i in range(len(tps)):
        precisions.append(computeBinaryPrecision(tps[i], fps[i], fns[i]))

    if precisions:
        precision = sum(precisions) / len(precisions)
    return precision

def computeMacroRecall(tps: list[int], fps: list[int], fns: list[int], data_size: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    recall = float(0)

    # Calculate recall for each class and average
    recalls = []
    for i in range(len(tps)):
        recalls.append(computeBinaryRecall(tps[i], fps[i], fns[i]))
    
    if recalls:
        recall = sum(recalls) / len(recalls)
    return recall

def computeMacroFMeasure(tps: list[int], fps: list[int], fns: list[int], data_size: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    f_measure = float(0)

    # Calculate F-measure for each class and average
    f_measures = []
    for i in range(len(tps)):
        f_measures.append(computeBinaryFMeasure(tps[i], fps[i], fns[i]))

    if f_measures:
        f_measure = sum(f_measures) / len(f_measures)
    return f_measure

def computeAccuracy(tps: list[int], fps: list[int], fns: list[int], data_size: int) -> float:
    # Have fun! Below is just a dummy for returns, feel free to edit
    accuracy = float(0)

    # Overall accuracy is sum of TPs divided by total data size
    if data_size > 0:
        accuracy = sum(tps) / data_size
    return accuracy

# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.

# INPUT: classified_data   : a numpy array containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures
def evaluateKNN(classified_data: numpy.typing.NDArray, confusion_func=confusionMatrix) \
        -> tuple[float, float, float, float]:
    # Have fun! Below is just a dummy for returns, feel free to edit
    precision = float(0)
    recall = float(0)
    f_measure = float(0)
    accuracy = float(0)

    # Calculate confusion matrix
    conf_matrix = confusion_func(classified_data)

    # Calculate TPs, FPs, FNs
    tps = computeTPs(conf_matrix)
    fps = computeFPs(conf_matrix)
    fns = computeFNs(conf_matrix)

    # Calculate data size (exclude header row)
    data_size = len(classified_data) - 1

    # Calculate evaluation metrics
    precision = computeMacroPrecision(tps, fps, fns, data_size)
    recall = computeMacroRecall(tps, fps, fns, data_size)
    f_measure = computeMacroFMeasure(tps, fps, fns, data_size)
    accuracy = computeAccuracy(tps, fps, fns, data_size)

    return precision, recall, f_measure, accuracy

##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################

# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Helper.parseArguments()
    if not opts:
        print("Missing input. Read the README file.")
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Helper.readCSVFile(opts['classified_data'])
    if classified_data.size == 0:
        print("Classified data is empty, cannot run evaluation. Exiting Task 2.")
        return
    print('Evaluating kNN')
    result = evaluateKNN(classified_data)
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))

if __name__ == '__main__':
    main()

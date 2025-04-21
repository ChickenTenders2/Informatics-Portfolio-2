# In this file please complete the following task:
#
# Task 3 [7 points out of 30] Cross validation
# On your own, evaluate your classifier using the k-fold cross-validation technique covered in the lectures
# (use the training data only). You need to implement the data partitioning and the training and testing data per
# round preparation functions on your own. Output the average precisions, recalls, F-measures and accuracies resulting
# from cross-validation by implementing the relevant function and incorporating what you have implemented in Task 1
# and Task 2 appropriately.
#
# You can rely on the dummy functions if you have not attempted these tasks, and during marking the code will be
# invoked against teacher-implemented versions of these tasks. When invoking functions from Tasks 1 and 2, rely only on
# those that were already defined in the template.
#
# You can start working on this task immediately. Please consult at the very least Week 3 materials.

import os
import Helper
import Task_1
import Task_2
import Dummy
import numpy
import Task_4
from typing import Callable

# This function takes in the data for cross evaluation and the number of partitions to split the data into.
# The input data contains the header row. The resulting partitions do not.

# INPUT: training_data     : a numpy array containing paths to images and actual classes (at the very least, but other
#                            columns can be ignored)
#                            Please refer to Task 1 for precise format description. Remember, this data contains
#                            header row!
#        f                 : the number of partitions to split the data into, value is greater than 0,
#                            not guaranteed to be smaller than data size.
# OUTPUT: partition_list   : a list of numpy arrays, where each array represents a partition, so a subset of entries
#                           of the original dataset s.t. all partitions are disjoint, roughly same size (can differ by
#                            at most 1), and the union of all partitions equals the original dataset minus header row.
#                           THE PARTITIONS DO NOT CONTAIN HEADER ROWS.

def partitionData(training_data: numpy.typing.NDArray, f: int) -> list[numpy.typing.NDArray]:
    # Have fun! Below is just a dummy for returns, feel free to edit
    partition_list = []

    # Get data excluding the header row
    data_without_header =  training_data[1:]

    # Shuffle the data to ensure random distribution across partitions
    # Create a copy to avoid modifying the original data
    numpy.random.shuffle(data_without_header.copy())

    # Calculate size of each partition
    total_data_size = len(data_without_header)
    base_partition_size = total_data_size // f
    extra = total_data_size % f # Remaining entries to distribute

    # Partitions created
    start_idx = 0
    for i in range(f):
        # Add an extra item to some partitions if needed to distribute all data
        current_partition_size = base_partition_size + (1 if i < extra else 0)
        end_idx = start_idx + current_partition_size

        # Create partition and add it to the partition list
        if start_idx < end_idx: # Make sure no empty partitions are created
            partition = data_without_header[start_idx:end_idx]
            partition_list.append(partition)

        start_idx = end_idx
    return partition_list

# This function transforms partitions into training and testing data for each cross-validation round (there are
# as many rounds as there are partitions); in other words, we prepare the folds.
# Please remember that the training and testing data for each round must include a header at this point.

# INPUT: partition_list     : a list of numpy arrays, where each array represents a partition (see partitionData function)
#        f                  : the number of folds to use in cross-validation, which is the same as the number of
#                             partitions the data was supposed to be split to, and the number of rounds in cross-validation.
#                             Value is greater than 0.
# OUTPUT: folds             : a list of 3-tuple s.t. the first element is the round number, second is the numpy array
#                             representing the training data for that round, and third is the numpy array representing
#                             the testing data for that round
#                             The round numbers START WITH 0
#                             You must make sure that the training and testing data are ready for use
#                             (e.g. contain the right headers already)

def preparingDataForCrossValidation(partition_list: list[numpy.typing.NDArray], f: int) \
        -> list[tuple[int, numpy.typing.NDArray, numpy.typing.NDArray]]:
    # This is just for error handling, if for some magical reason f and number of partitions are not the same,
    # then something must have gone wrong in the other functions and you should investigate it
    if len(partition_list) != f:
        print("Something went really wrong! Why is the number of partitions different from f??")
        return []
    # Defining the header here for your convenience
    header = numpy.array([["Path", "ActualClass"]])
    folds = []

    # Implement your code here
    # For each partition, create a fold where the current partition is testing data, and the rest of the partitions are training data
    for i in range(f):
        # Current partition = testing data
        testing_data = numpy.vstack((header, partition_list[i]))

        # All other partitions = training data
        training_partitions = []
        for j in range(f):
            if j != i: # Skip the current testing partition
                training_partitions.append(partition_list[j])

        # Combine all training partitions
        if training_partitions:
            combined_training = numpy.vstack(training_partitions)
            training_data = numpy.vstack((header, combined_training))
        else:
            # If there's only one partition
            training_data = header

        # Add the fold to the list
        folds.append((i, training_data, testing_data))
    return folds

# This function takes the classified data from each cross validation round and calculates the average precision, recall,
# accuracy and f-measure for them.
# Invoke either the Task 2 evaluation function or the dummy function here, do not code from scratch!
#
# INPUT: classified_data_list
#                           : a list of numpy arrays representing classified data computed for each cross validation round
#        evaluation_func    : the function to be invoked for the evaluation (by default, it is the one from
#                             Task_2, but you can use dummy)
# OUTPUT: avg_precision, avg_recall, avg_f_measure, avg_accuracy
#                           : average evaluation measures. You are expected to evaluate every classified data in the
#                             list and average out these values in the usual way.

def evaluateResults(classified_data_list: list[numpy.typing.NDArray], evaluation_func=Task_2.evaluateKNN) \
        -> tuple[float, float, float, float]:
    avg_precision = float(0)
    avg_recall = float(0)
    avg_f_measure = float(0)
    avg_accuracy = float(0)
    # There are multiple ways to count average measures during cross-validation. For the purpose of this portfolio,
    # it's fine to just compute the values for each round and average them out in the usual way.

    # Check for any classified data to be evaluated
    if not classified_data_list:
        return avg_precision, avg_recall, avg_f_measure, avg_accuracy
    
    precisions = []
    recalls = []
    f_measures = []
    accuracies = []

    # Evaluate each classified dataset
    for classified_data in classified_data_list:
        if len(classified_data) <= 1: # Skip any empty or header-only data
            continue

        # Get evaluation metrics for this fold
        precision, recall, f_measure, accuracy = evaluation_func(classified_data)

        # Add to collection
        precisions.append(precision)
        recalls.append(recall)
        f_measures.append(f_measure)
        accuracies.append(accuracy)

    # Compute averages if there are valid evaluations
    if precisions:
        avg_precision = sum(precisions) / len(precisions)
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
    if f_measures:
        avg_f_measure = sum(f_measures) / len(f_measures)
    if accuracies:
        avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_precision, avg_recall, avg_f_measure, avg_accuracy

# In this task you are expected to perform and evaluate cross-validation on a given dataset.
# You are expected to partition the input dataset into f partitions, then arrange them into training and testing
# data for each cross validation round, and then run kNN for each round using this data and k, measure_func, and
# similarity_flag that are provided at input (see Task 1 for kNN input for more details).
# The results for each round are collected into a list and then evaluated.
#
# You are then asked to produce an output dataset which extends the original input training_data by adding
# "PredictedClass" and "FoldNumber" columns, which for each entry state what class it got predicted when it
# landed in a testing fold and what the number of that fold was (everything is in string format).
# This output dataset is then extended by two extra rows which add the average measures at the end.
#
# You are expected to invoke the Task 1 kNN classifier or the Dummy classifier here, do not implement these things
# from scratch! You must use the other relevant function defined in this file.
#
# INPUT: training_data      : a numpy array that was read from the training data csv (see parse_arguments function)
#        f                  : the number of folds, greater than 0, not guaranteed to be smaller than data size.
#        k                  : the value of k neighbours, greater than 0, not guaranteed to be smaller than data size.
#        measure_func       : the function to be invoked to calculate similarity/distance (see Task 4 for
#                               some teacher-defined ones)
#        similarity_flag    : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#        knn_func           : the function to be invoked for the classification (by default, it is the one from
#                             Task_1, but you can use dummy)
#        partition_func     : the function used to partition the input dataset (by default, it is the one above)
#        prep_func          : the function used to transform the partitions into appropriate folds
#                            (by default, it is the one above)
#        eval_func          : the function used to evaluate cross validation (by default, it is the one above)
# OUTPUT: output_dataset    : a numpy array which extends the original input training_data by adding "PredictedClass"
#                             and "FoldNumber" columns, which for each entry state what class it got predicted when it
#                             landed in a testing fold and what the number of that fold was (everything is in string
#                             format). This output dataset is then extended by two extra rows which add the average
#                             measures at the end (see the h and v variables).
def crossEvaluateKNN(training_data: numpy.typing.NDArray, f: int, k: int, measure_func: Callable,
                     similarity_flag: bool, knn_func=Task_1.kNN,
                     partition_func=partitionData, prep_func=preparingDataForCrossValidation,
                     eval_func=evaluateResults) -> numpy.typing.NDArray:
    # This adds the header
    output_dataset = numpy.array([['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']])
    avg_precision = -1.0;
    avg_recall = -1.0;
    avg_fMeasure = -1.0;
    avg_accuracy = -1.0;
    classified_list = []

    # Have fun with the computations!
    # Partition the data
    partitions = partition_func(training_data, f)

    #Prepare folds for cross-validation
    folds = prep_func(partitions, f)

    # Run kNN on each fold and collect results
    for fold_num, train_data, test_data in folds:
        # Run kNN classifier
        classified_data = knn_func(train_data, test_data, k, measure_func, similarity_flag)
        classified_list.append(classified_data)

        # Add fold number to the classified data
        for i in range(1, len(classified_data)): # Skipped header
            # Extract data from the classified result
            path = classified_data[i][0]
            actual_class = classified_data[i][1]
            predicted_class = classified_data[i][2]

            # Add fold number to output dataset
            output_dataset = numpy.append(output_dataset, [[path, actual_class, predicted_class, str(fold_num)]], axis=0)

    # Evaluate the results
    avg_precision, avg_recall, avg_fMeasure, avg_accuracy = eval_func(classified_list)

    # The measures are now added to the end.
    h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall, avg_fMeasure, avg_accuracy]

    output_dataset = numpy.append(output_dataset, [h], axis=0)
    output_dataset = numpy.append(output_dataset, [v], axis=0)

    return output_dataset

##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################

# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Helper.parseArguments()
    if not opts:
        print("Missing input. Read the README file.")
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Helper.readCSVFile(opts['training_data'])
    if training_data.size == 0:
        print("Input data is empty, cannot run cross-validation. Exiting Task 3.")
        return
    if opts['f'] is None or opts['f'] < 1:
        print("Value of f is missing from input or too small, cannot run cross validation. Exiting Task 3.")
        return
    if opts['k'] is None or opts['k'] < 1:
        print("Value of k is missing from input or too small, cannot run cross validation. Exiting Task 3.")
        return
    if opts['simflag'] is None:
        print("Similarity flag is missing from input, cannot run cross validation. Exiting Task 3.")
        return
    print('Running cross validation')

    try:
        result = crossEvaluateKNN(training_data, opts['f'], opts['k'], eval(opts['measure']), opts['simflag'])
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

    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Helper.writeCSVFile(out, result)

if __name__ == '__main__':
    main()
##########################################################################################
# Task 4 [5 points out of 30] Similarities
#
# Independent inquiry time! In Task 1, you were instructed to use similarity and distance measures that were prepared
# by the teacher beforehand. Now it’s time to implement your own.
#
# On your own, implement the RGB image versions of cosine similarity and RMSE distance. You can use functions for basic
# image transformations (such as flattening) and you can use functions for calculating the root or power.
# The remaining elements, such as dot product, average, sum, etc., need to be implemented on your own.
#
# You can start working on this task immediately. Please consult at the very least Week 1 materials.
##########################################################################################

import numpy
import sewar
from math import sqrt, pow
import Helper

# This function computes cosine similarity between two images.
# These images are as read from Helper.readAndResize function and are in RGB format.
# Use appropriate type transformation as needed.
# Do not transform to grayscale. Do not remove channels.
# INPUT:
#       image1, image2: two images to compare
# OUTPUT:
#       value : the similarity value between the images (as float)
def computeCosineSimilarity(image1: numpy.typing.NDArray, image2: numpy.typing.NDArray) -> float:
    # Remember: the images are in RGB format. DO NOT TRANSFORM THEM TO GRAYSCALE.
    # Flaten the images to 1D arrays while keeping RGB values
    flat_img1 = image1.flatten().astype(float)
    flat_img2 = image2.flatten().astype(float)

    # Calculate dot product manually
    dot_product = 0.0
    for i in range(len(flat_img1)):
        if i < len(flat_img2): # Prevent exceeding bounds of either array
            dot_product += float(flat_img1[i]) * float(flat_img2[i])

    # Calculate magnitudes manually
    magnitude1 = 0.0
    for pixel in flat_img1:
        magnitude1 += float(pixel) * float(pixel)
    magnitude1 = sqrt(magnitude1)

    magnitude2 = 0.0
    for pixel in flat_img2:
        magnitude2 += float(pixel) * float(pixel)
    magnitude2 = sqrt(magnitude2)

    # Avoid dividing by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

# This function computes RMSE distance between two images.
# These images are as read from Helper.readAndResize function and are in RGB format.
# Use appropriate type transformation as needed.
# Do not transform to grayscale. Do not remove channels.
# INPUT:
#       image1, image2: two images to compare
# OUTPUT:
#       value : the distance value between the images (as float)

def computeRMSEDistance(image1: numpy.typing.NDArray, image2: numpy.typing.NDArray) -> float:
    # Remember: the images are in RGB format. DO NOT TRANSFORM THEM TO GRAYSCALE.
    # Flatten
    flat_img1 = image1.flatten().astype(float)
    flat_img2 = image2.flatten().astype(float)

    # Determine minimum length to avoid index errors
    min_length = min(len(flat_img1), len(flat_img2))

    # Calculate sum of squared diferences manually
    sum_squared_diff = 0.0
    for i in range(min_length):
        diff = float(flat_img1[i]) - float(flat_img2[i])
        sum_squared_diff += diff * diff

    # Mean squared error
    if min_length > 0:
        mse = sum_squared_diff / min_length
    else:
        mse = 0.0

    # Root mean squared error
    rmse = sqrt(mse)
    return rmse

##########################################################################################
# Here is a teacher-defined similarity function you can use for Task 1
# DO NOT OVERRIDE
def computePSNRSimilarity(image1: numpy.typing.NDArray, image2: numpy.typing.NDArray) -> float:
    return sewar.psnr(image1,image2)

def main():
    opts = Helper.parseArguments()
    if not opts:
        print("Missing input. Read the README file.")
        exit(1)
    print(f'Reading images from {opts["image_a"]} and {opts["image_b"]}')
    image_a = Helper.readAndResize(opts['image_a'])
    image_b = Helper.readAndResize(opts['image_b'])
    if image_a.size == 0 or image_b.size == 0:
        print("Not all of the images have been read correctly, cannot calculate measures. Exiting Task 4.")
        return

    try:
        measure_name = opts['measure']
        measure = eval(measure_name)
        result = measure(image_a,image_b)
    except NameError as nerror:
        print(nerror)
        print("Wrong measure function name was passed to the function, please double check the function name. "
              "For example, try 'computePSNRSimilarity' instead of 'Task_4.computePSNRSimilarity'.")
        return
    except TypeError as terror:
        print("Measure function is incorrect or missing, please double check the input. "
              "For example, try 'Task_4.computePSNRSimilarity' and make sure you have not deleted any imports "
              "from the template.")
        return
    print("Measure "+ str(measure_name)+" between the images results in "+str(result))

if __name__ == '__main__':
    main()
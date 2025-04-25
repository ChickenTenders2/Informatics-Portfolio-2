##########################################################################################
# Task 5 [5 points out of 30] I can do better!
#
# Independent inquiry time! There are much better approaches out there for image classification.
# Your task is to find one and analyse it:
# •	State the name of the approach
# •	Provide a permalink to a resource in the Cardiff University library that describes the approach AND
#   include an appropriate reference
# •	Briefly explain how the approach you found is better than kNN in image classification.
#   Focus on synthesis, not recall!
#
# The reference can be in any format you wish but has to contain all necessary components.
# You can start working on this task immediately. Please consult at the very least Week 2 materials.
##########################################################################################

"""
Intelligent Systems and Sustainable Computing (2023)
Chapter 14: SVM Versus KNN: Prediction of Best Image Classifier

Permalink: Permalink: https://librarysearch.cardiff.ac.uk/discovery/fulldisplay?docid=cdi_springer_books_10_1007_978_981_99_4717_1_14&context=PC&vid=44WHELF_CAR:44WHELF_CAR_VU1&lang=en&search_scope=CU_Search_ALL&adaptor=Primo%20Central&tab=CSCOP_EVERYTHING&query=any,contains,image%20classifier&offset=10

Reference: Marri, S. P., Nikith, B. V., Keerthan, N. K. S., & Jayan, S. (2023). SVM Versus KNN: 
Prediction of Best Image Classifier. In V. S. Reddy, V. K. Prasad, J. Wang, & N. M. R. Dasari 
(Eds.), Intelligent Systems and Sustainable Computing: Proceedings of ICISSC 2022 
(pp. 147-159). Springer Nature Singapore.

Through my research in Cardiff University's library, I discovered that support vector machines (SVMs) significantly outperform the
k-Nearest Neighbors (kNN) approach I implemented earlier. SVMs use optimal hyperplane seperation with maximum margins, enhancing performance
in high-dimensional image classification problems.

While kNN relies on proximity-based classification and struggles with large datasets, SVMs transform nonlinear data through specialized
kernel functions, achieving 86% accurancy on human/animal classification versus kNN's 69%.

SVM's advantages include better classification margins and reduced tendency to memorize training data, making them valuable for complex
images where feature relationships extend beyond basic distance measures. 

"""
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

Through my research in Cardiff University's library, I discovered that support vector machines (SVMs) have a significant advancement
over the k-Nearest Neighbors (kNN) approach I implemented in earlier tasks. SVMs implement optimal hyperplane separation with maximum
margins, significantly enhancing its ability to handle the high-dimensional feature spaces encountered in real-world image classification
problems. The libary resource I chose provides evidence for why more advanced techniques exists and are needed beyond kNN implementations.

While kNN relies on proximity-based classification and struggles with computational efficiency while dealing with large datasets such as
the ones used for previous tasks, SVMs transform nonlinear data through specialized kernel functions (e.g. poly, RBF) to create separable
representations. This difference allows SVMs to achieve 86% accuracy on human/animal classification compared to kNN's 69%, proving SVM's
generalization capabilities and consistent robust results across a large and diverse image dataset. However, there are occasions where 
kNN outperforms the SVMs, in specific contexts such as breast cancer detection with 98% accuracy. 

SVM's advantages extend beyond accuracy to include better classification margins and reduced resistance to memorizing training data. 
These qualities make SVMs valuable for complex image classication tasks where feature relationships extend beyond basic distance measures,
which is a key limitation of kNN's distance-based approach. 

"""
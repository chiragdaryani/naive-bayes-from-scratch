# Intro to NLP - Assignment 5

In this assignment, our aim is to build a Naive Bayes model based on bag-of-word (BoW) features to classify the relation of a sentence. The program processes and classifies a sentence as indicating one of the following relations: publisher, director, performer, and characters.

## Team
|Student name| CCID |
|------------|------|
|Chirag Daryani   |  cdaryani    |
|Varshini Prakash   |  vprakash    |



## 3-rd Party Libraries
Apart from `nltk` and `pandas`, we have used the following 3rd party libraries:

* `main.py L:258, L:395` used **`scikit-learn accuracy_score function`** for calculating the accuracy.
* `main.py L:262` used **`scikit-learn confusion_matrix function`** for making the confusion matrix for the model predictions.
* `main.py L:289` used **`scikit-learn precision_recall_fscore_support function`** for calculating the precision, recall for each class label.
* `main.py L:303-308` used **`scikit-learn precision_score, recall_score functions`** for calculating the Micro-Macro Precision/Recall.
* `main.py L:342` used **`scikit-learn KFold function`** for getting the object for implementing KFold Cross Validation.


## How to Execute?

To run this project,

1. Download the repository as a zip file.
2. Extract the zip to get the project folder.
3. Open Terminal in the directory you extracted the project folder to. 
4. Change directory to the project folder using:

    `cd f2021-asn5-chiragdaryani-main`
   
5. Install the required libraries, **Pandas** and **scikit-learn** using the following commands:

    `pip3 install pandas`

    `pip3 install -U scikit-learn`
 
6. Now to execute the code, use the following commands (in the current directory):

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`


## Description of the execution command

Our program **src/main.py** takes three command-line options. The first is the path to the training file, the second is the path to the test file and 
the third is the path to the output file. 

The assignment's training data can be found in [data/train.csv](data/train.csv),and the in-domain test data can be found in [data/test.csv](data/test.csv).

The output file must be generated in the [output/](output/) directory.

So specifying these paths, one example of a possible execution command is:

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`


## References

https://theflyingmantis.medium.com/text-classification-in-nlp-naive-bayes-a606bf419f8c

https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece

https://github.com/vamc-stash/Naive-Bayes/blob/master/src/GaussianNB.py

https://github.com/yasharkor/relation-extraction-classification/blob/98b7271399ae5fa4ececbc78568a1f107e647a3c/code/train.py#L45

https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826

https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/

https://stackoverflow.com/questions/35178590/scikit-learn-confusion-matrix

https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right

https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string

## Note

We are using the updated dataset (v2) for this project.

           

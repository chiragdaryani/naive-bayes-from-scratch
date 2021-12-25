import csv
import argparse

from statistics import mean

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold

import pandas as pd


# Initialize the argument parser
parser = argparse.ArgumentParser()

# Add the parameters we will pass from cli
parser.add_argument('--train',help='path to the input training data file')
parser.add_argument('--test',help='path to the input test data file')
parser.add_argument('--output',help='path to the output file')

# Parse the arguments
args = parser.parse_args() 
#print(args)

# Path to input training data file
input_train_data_path= args.train
# Path to input test data file
input_test_data_path= args.test
# Path to output file
output_file_path= args.output





'''
Function to read data from input file, clean the data, get all individual words in each sentence, get label for each sentence
'''

def pre_process_data(file_path):

    try:


        # Read input from "train.tsv" or "test.csv" data file
        with open(file_path, 'r') as input_csv_file:
                
                input_data_file = csv.reader(input_csv_file)
                # skip the first line i.e headers (column names)
                next(input_data_file)
                
                labels=[]
                word_list_per_sentence=[]
                row_ids=[]

                # Go through each sentence one by one
                for row in input_data_file:

                    row_id = row[0]
                    row_ids.append(row_id)
                    
                    label= row[2]
                    labels.append(label)

                    sentence_words=[]

                    #splitting full sentence based on space
                    sentence= row[1].split(" ")
                    for word in sentence:
                        #remove puntuations, special characters/ symbols 
                        if word.isalnum():
                            sentence_words.append(word.lower())

                    word_list_per_sentence.append(sentence_words)

                #print(word_list_per_sentence) #words in each sentence
                #print(labels) #class of each sentence
                
                return word_list_per_sentence, labels, row_ids
    
    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)

                






'''
Function to Train the Naive Bayes Classifier by calculating all probabilities as per the formula of Naive Bayes Classfier
'''

def train_NaiveBayes(word_list_per_sentence, labels, vocab):

    try:


        vocab_size = len(vocab)
        

        '''
        Calculate Probability of Each Class P(y)
        '''
        
        probab_per_class={} #To store Probability of each class P(y)

        for i in range(len(labels)):

            if labels[i] in probab_per_class.keys(): #class label found in dictionary, so increment its count
                probab_per_class[labels[i]]= probab_per_class[labels[i]]+1
        
            else: #class label not found in dictionary, so give it a count of 1
                probab_per_class[labels[i]] = 1

        #Divide by total no of labels to convert count of each class to probability of each class
        for key in probab_per_class.keys():
            probab_per_class[key]= probab_per_class[key] / len(labels)





        '''
        Calculate Probability of a Word given it belongs to a Class P( xi | y=class k)
        '''

        #List of ALL words (in ALL SENTENCES) for each class label
        label_and_words_in_sentence= list(zip(labels, word_list_per_sentence))

        #Let's also make a dictionary with key as class label, each word in that class and its count (in ALL sentences of that class) as value
        each_label_all_words={}

        #Let's track count of words (in ALL SENTENCES) for each class label
        each_label_word_counts = {} 
        
        #Intialize the dictionaries
        for ele in label_and_words_in_sentence:
            each_label_all_words[ele[0]]={}
            each_label_word_counts[ele[0]]=0
        
        # Put values into these dictionaries
        for ele in label_and_words_in_sentence:
            for word in ele[1]:
                each_label_word_counts[ele[0]]= each_label_word_counts[ele[0]]+1

                if word in each_label_all_words[ele[0]]:
                    each_label_all_words[ele[0]][word]=each_label_all_words[ele[0]][word]+1

                else:
                    each_label_all_words[ele[0]][word]=1


        #WE NOW HAVE EACH LABEL AND ALL THE WORDS THAT WERE ENCOUNTERED IN SENTENCES FOR THAT LABEL (WITH THEIR COUNTS)

        #Now we have all values to calculate probability of word given class i.e P(xi | y) 

        word_probab_given_class={} #dictionary to store probab of word given it belongs to sentence of given class
        #It will be a 2D dictionary with key as class label, value will be of the form 'word': probab of word given it belongs to class

        # The probability of word i given class j is the count that the word occurred in sentences of class j, 
        # divided by the sum of the counts of each word in our vocabulary in class j
        
        for class_label in each_label_all_words.keys():

            #intitialise dictionary to store probab of words for given class
            word_probab_given_class[class_label]={}

            #initialize probability for unknown symbol
            word_probab_given_class[class_label]['<UNK>'] = 1 / each_label_word_counts[class_label]

            #calculate probability for all other words given class label
            for word in each_label_all_words[class_label].keys():

                '''
                Laplace Smoothing: Add 1 in numerator, Vocab size in Denominator
                '''
                word_probab_given_class[class_label][word] = (each_label_all_words[class_label][word]+1) / (each_label_word_counts[class_label]+vocab_size)
            

        return probab_per_class, word_probab_given_class
    


    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)







'''

We will apply the Naive Bayes formula to calculate P ( class_label / sentence )
We then find which class is giving max value of above probability
That class will be our prediction.

'''
def get_NaiveBayes_prediction(sentence,probab_per_class, word_probab_given_class, train_vocab ):

    try:

        prediction_probabilities={}

        # Get class labels for which we want to calculate prediction probability
        class_labels=[]
        for key in probab_per_class.keys():
            class_labels.append(key)

        # Go through each class and calculate prediction probability for sentence to belong to that class
        for class_label in class_labels:
    
            probab =1 #intialization

            for word in sentence:

                #If the word is not present in the vocabulary created using train sentences, then we must ignore such words
                if word not in train_vocab:
                    continue
                
                #If word not in the words for that class, use the UNKNOWN token probability for the word
                if word not in word_probab_given_class[class_label].keys():
                    word_probab= word_probab_given_class[class_label]['<UNK>']
                
                else:
                    word_probab= word_probab_given_class[class_label][word]
                
                probab = probab * word_probab
            
            #finally multiply with P(class_label) to get complete probability prediction for the class
            probab = probab * probab_per_class[class_label]

            #append to dict of final prediction probabilities for each class
            prediction_probabilities[class_label]=probab
        
        #find class wih max prediction probability 
        predicted_class= max(prediction_probabilities, key=prediction_probabilities.get)
        return predicted_class

    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)








'''

Function to calculate Accuracy, Precision, Recall and Make Confusion Matrix

'''
        
def calculatePerformanceMetrics(y_test, y_pred):

    try:

        print('\nAccuracy on Test Dataset: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))

        # Creating  a confusion matrix,which compares the y_test and y_pred
        print('\nConfusion Matrix:\n')
        cm = confusion_matrix(y_test, y_pred, labels=[ 'director', 'publisher', 'performer','characters'])
            
        # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
        cm_df = pd.DataFrame(cm,
                        index = ['director', 'publisher', 'performer','characters'], 
                        columns = ['director', 'publisher', 'performer','characters'])
        
        #print(cm_df)

        '''
        Note: According to sklearn, actual class values are represented by rows of this confusion matrix and predicted class values are represented as
        columns of this confusion matrix. 

        Now according to the requirement specified in the assignment specification, the final confusion matrix must have actual class values represented
        as columns and predicted class values represented as rows.

        So  we will take transpose of this confusion matrix
        '''

        print(cm_df.transpose())
        print("")


        '''
        Get Precision, Recall of each class
        '''
        
        precision, recall, fscore, support = score(y_test, y_pred, labels=['director', 'publisher', 'performer','characters'])
        
        labels=['director', 'publisher', 'performer','characters']

        #print('Precision: {}'.format(precision))
        #print('Recall: {}'.format(recall))
        
        print(''.join('Precision for class {}:{:.4f}\n'.format(first, second) for first, second in zip(labels, precision)))
        print(''.join('Recall for class {}:{:.4f}\n'.format(first, second) for first, second in zip(labels, recall)))

        print('\nClassification Report:\n')
        print(classification_report(y_test, y_pred, labels=['director', 'publisher', 'performer','characters'], digits=4))

        print("\n")
        print('Micro Average Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
        print('Micro Average Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
        print("\n")

        print('Macro Average Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Average Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))



    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)


    





'''
Function to perform K-Fold Cross Validation (K=3)
'''
    
    
def three_fold_cross_eval():

    try:


        print("\n3-fold Cross Eval Started!\n")


        word_list_per_sentence=[]
        labels=[]

        #read training data    
        word_list_per_sentence, labels, train_rowids= pre_process_data(input_train_data_path)

        #KFold Object with K=3
        kf = KFold(n_splits=3)



        kfold_accuracies=[] #this will store the accuracy calcualated on held out test data in each of the 3 folds

        for train_index, test_index in kf.split(word_list_per_sentence):

            #get indexes that will be used for train and indexes that will be used for test
            #print("TRAIN:", train_index, "TEST:", test_index)

            '''
            Using these indexes, get your training and testing dataset
            '''

            train_sentences=[]
            for index in train_index:
                train_sentences.append(word_list_per_sentence[index])

            test_sentences=[]
            for index in test_index:
                test_sentences.append(word_list_per_sentence[index])
            
            train_labels=[]
            for index in train_index:
                train_labels.append(labels[index])

            test_labels=[]
            for index in test_index:
                test_labels.append(labels[index])

            
            # Create Vocabulary for training sentences
            train_vocab = set([word for sentence in train_sentences for word in sentence])
           

        
            '''Now train Naive Bayes Model on these train sentences'''

            probab_per_class, word_probab_given_class= train_NaiveBayes(train_sentences, train_labels, train_vocab)


            '''Now we must get predictions from this model on test sentences, calculate accuracy'''

            predicted_labels=[]
            for sentence in test_sentences:
                prediction = get_NaiveBayes_prediction(sentence, probab_per_class, word_probab_given_class, train_vocab)
                predicted_labels.append(prediction)

            

            #Calculate Accuracy 
            accuracy = accuracy_score(test_labels, predicted_labels)

            kfold_accuracies.append(accuracy)
        
        print("The accuracies on each test hold out set:",kfold_accuracies)

        avg_accuacy = mean(kfold_accuracies)
        #print(avg_accuacy)
        return avg_accuacy
    
    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)








'''
Function to write the predictions to the output file
'''
def write_predictions_output_file(test_rowids, test_true_labels, predicted_labels):

    
    try:

        # create the output csv file in which we will write our final results
        with open(output_file_path, 'w', newline='') as outputFile:

                writer = csv.writer(outputFile)
                
                # create headers for output file
                writer.writerow(["original_label", "output_label","row_id"])

                zipped_output = list(zip(test_true_labels, predicted_labels, test_rowids))

                #Now we will write these values to the output file
                writer.writerows(zipped_output)  

    except Exception as e:
        print("Something went wrong!\n The exception message is: ",e)








            



def main():

    try:

       word_list_per_sentence=[]
       labels=[]
       word_list_per_sentence, labels, train_rowids= pre_process_data(input_train_data_path)
       
       
       #Convert 2D List of Sentences into 1D list of words to calculate Vocabulary size       
       flattened_list = [word for sentence in word_list_per_sentence for word in sentence]
       #Get unique words only
       vocab=set(flattened_list)
       #vocab_size = len(vocab)

       #print(vocab)

       





       '''
       Call Function to do 3-fold cross validation on train data, Get Average Accuracy as result    
       '''
       
       average_3fold_accuracy= three_fold_cross_eval()
       print("The Average Training Accuracy using 3-Fold Cross Validation is :{:.4f}".format(average_3fold_accuracy))









       '''
       CREATING NAIVE BAYES MODEL ON FULL TRAINING DATA
       '''

       probab_per_class, word_probab_given_class= train_NaiveBayes(word_list_per_sentence, labels, vocab)


       



       
       
       #Now we will read test sentences, preprocess them
       test_sentences, test_true_labels, test_rowids= pre_process_data(input_test_data_path)

       '''
       Get Predictions on Test Data
       '''

       predicted_labels=[]
       for sentence in test_sentences:
           prediction = get_NaiveBayes_prediction(sentence, probab_per_class, word_probab_given_class, vocab)
           predicted_labels.append(prediction)


       '''
       Call Function to write predictions to output file
       '''

       write_predictions_output_file(test_rowids, test_true_labels, predicted_labels)


       '''
       Call Function to calculate all Performance Metrics on Test Data    
       '''

       print("\n\nEvaluation on test.csv started:")
       calculatePerformanceMetrics(test_true_labels, predicted_labels)


        
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()








#References:

#https://theflyingmantis.medium.com/text-classification-in-nlp-naive-bayes-a606bf419f8c
#https://github.com/vamc-stash/Naive-Bayes/blob/master/src/GaussianNB.py
#https://github.com/yasharkor/relation-extraction-classification/blob/98b7271399ae5fa4ececbc78568a1f107e647a3c/code/train.py#L45
#https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
#https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
#https://stackoverflow.com/questions/35178590/scikit-learn-confusion-matrix
#https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right
#https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
#https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
           
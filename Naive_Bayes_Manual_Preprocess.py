#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from collections import Counter
import re


class Naive_Bayes:
    def __init__(self, X_train, Y_train):
        ##################################################################
        ####Initialize the class with the Training features and labels####
        ##################################################################
        self.X_train = X_train
        self.Y_train = Y_train
        self.classes = np.unique(Y_train)
        self.words_freq_per_class, self.no_unique_words_per_class, self.total_word_count_per_class = self.add_to_bow_dict()
        
    
    def process_abstract(self, abstract):
        ######################################################
        ####Preprocess the abstracts##########################
        ####Input - each abstract text########################
        ####Output - returns list of words after processing###
        ######################################################
        remove_words = ['my','myself','we','our','ours', 'ourselves','you','your','yours',
                        'yourselves','he','him','his', 'himself','she','her','hers','herself', 'yourself',
                        'it','its','itself','they','them','their','theirs','themselves','what', 'further',
                        'which','who','whom','this','that','these','those','am','is','are','was','were','be',
                        'been','being','have','has','had', 'having','do','does','did','doing','a','an','the', 
                        'and','but','if','or','because','as','until','while','of','at','by','for', 'with',
                        'about','against','between','into','through','during','before','after', 'above',
                        'below','to','from','up','down','in','out', 'on', 'off','over','under','again',
                        'then','once','here','there','when', 'where','why','how','all','any','both','each',
                        'few','more','most','other','some','such','no','nor','not','only','own','same',
                        'so','than','too','very','can','will','just','should','now']
        cleaned_str=re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", abstract)#remove hyperlinks
        cleaned_str=re.sub(r"(\$+)(?:(?!\1)[\s\S])*\1", "", cleaned_str)#remove latex between $$
        cleaned_str=re.sub('[^a-z\s]+',' ',cleaned_str,flags=re.IGNORECASE) #remove symbols
        pat = r'\b(?:{})\b'.format('|'.join(remove_words))#remove useless words from the abstract
        cleaned_str = re.sub(pat, ' ', cleaned_str)
        cleaned_str=re.sub('(\s+)',' ',cleaned_str) #remove multiple spaces
        cleaned_str=cleaned_str.lower() #convert to lower case
        return cleaned_str.split() #return as lists
    
    def add_to_bow_dict(self):
        #################################################
        ####Computes Bag Of Words########################
        #################################################
        words_freq_per_class, no_unique_words_per_class, total_word_count_per_class = {}, {}, {}
        for clas in self.classes:
            
            # accumulate the list of words in the particular class
            word_list = []
            for abstract in self.X_train[self.Y_train == clas]:
                word_list += self.process_abstract(abstract)

            # count the number of unique words
            no_unique_words_per_class[clas] = len(set(word_list))
            
            #count of occurence each word / class
            words_freq_per_class[clas] = Counter(word_list)

            #total word count per class
            total_word_count_per_class[clas] = len(word_list)
        return words_freq_per_class, no_unique_words_per_class, total_word_count_per_class
    
    def compute_log_priors(self):
        ##############################################
        ####Computes Log Priors for each class########
        ##############################################
        
        priors ={}
        # compute log prior for all classes
        for clas in self.classes:
            priors[clas] = np.log(np.count_nonzero(self.Y_train == clas)/ len(self.Y_train))
        self.priors = priors
    
    def compute_probability(self, clas, word_list_per_class):
        #####################################################
        ####Computes Probability of class given words########
        #####################################################
        
        #taking log of probabilities to convert to addition. Anyway the final prediction is argmax
        probability=self.priors[clas]
        
        total_unique_vocab = sum(self.no_unique_words_per_class.values())
        
        for word in word_list_per_class:
            #[count(w|c)+1 ] / [ count(c) + |V|] by using add 1 smoothing
            denom = self.total_word_count_per_class[clas] + total_unique_vocab 
            probability += np.log(((self.words_freq_per_class[clas][word]) + 1)/denom) # +1 added as smoothing param alpha
        return probability
    
    def predict(self, array_abstract):
        ########################################################
        ####Predicts the class for the array of abstract########
        ########################################################
        
        #first get the priors of each class
        self.compute_log_priors()
        #print("priors", self.priors)
        
        #prediction of each abstract
        predicted_class = np.zeros(len(array_abstract), dtype=np.object)
        
        for i, abstract in enumerate(array_abstract):
            # same preprocessing as in train set
            words_in_abstract = self.process_abstract(abstract) 
            
            probs = np.zeros(len(self.classes)) 
            
            for index, clas in enumerate(self.classes):
                #compute word probabilities given class
                probs[index] = self.compute_probability(clas, words_in_abstract)

            predicted_class[i] = self.classes[np.argmax(probs)]
        
        return predicted_class


if __name__ == '__main__':
    
    #read train set
    data = pd.read_csv("train.csv")
    data = data.drop_duplicates()
    
    #split train set and test set for validation
    train_set = data[data.index < 0.8*(data.shape[0])].copy()
    val_set = data[data.index >= 0.8*(data.shape[0])].copy()
    
    #creating Naive bayes model
    model = Naive_Bayes(train_set['Abstract'].values, train_set["Category"].values)
    print("_____Training done_____")
    
    #validate the model and find the accuracy
    validation = model.predict(val_set['Abstract'].values)
    print("_____Validation done___")
    print("Accuracy on validation set", np.mean(validation == val_set['Category'].values))
    
    print("Predicting on test set......")
    #read test set
    test_set = pd.read_csv("test.csv")
    test_set = test_set.drop_duplicates()
    
    #predict on test set
    predictions = model.predict(test_set['Abstract'].values)
    
    #ouput the predictions on a csv file
    final_prediction = pd.DataFrame(predictions, index = np.arange(predictions.size))
    final_prediction.columns = ['Category']
    final_prediction['Id']=np.arange(final_prediction.size)
    final_prediction = final_prediction[["Id", "Category"]]

    final_prediction.to_csv(r'predictionsNB.csv', index=False)
    print("____O/P file created___")
    
    #Resources used
    #https://github.com/notAlex2/reddit_comment_classification/blob/master/naive_bayes.py
    #https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf





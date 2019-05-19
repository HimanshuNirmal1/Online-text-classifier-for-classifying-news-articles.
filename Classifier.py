'''
-Team #7
-An Online Web-Based Application for News Classification Based on Category Using Unique Bag of Words.
-Sachin Mohan Sujir, Himanshu Nirmal
'''


import pandas as pd
import numpy as np
from collections import defaultdict
import re

from nltk.stem import PorterStemmer


cachedStopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

ps = PorterStemmer()
def preprocess_string(string):

    cleaned_str=re.sub('[^a-z\s]+',' ',string,flags=re.IGNORECASE)
    cleaned_str=re.sub('(\s+)',' ',cleaned_str)
    cleaned_str=cleaned_str.lower()

    cleaned=ps.stem(cleaned_str)

    cleaned_str1= ' '.join([word for word in cleaned.split() if word not in cachedStopWords])


    return cleaned_str1


class MultinomialNaiveBayes:

    def __init__(self,unique_classes):

        self.classes=unique_classes


    def bagOfWords(self,headline,dict_index):


        if isinstance(headline,np.ndarray): headline=headline[0]

        for token_word in headline.split(): #each word in preprocessed string

            self.bow_dicts[dict_index][token_word]+=1 #increment in its count

    def train(self,dataset,labels):

        self.headline=dataset
        self.category=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])


        if not isinstance(self.headline,np.ndarray): self.headline=np.array(self.headline)
        if not isinstance(self.category,np.ndarray): self.category=np.array(self.category)

        #Developing Bag of Words for each category
        for cat_index,cat in enumerate(self.classes):

            all_cat_headline=self.headline[self.category==cat] #filter all headline of category == cat

            #removing stopwords,tokenizing headlines

            cleaned_headline=[preprocess_string(cat_headline) for cat_headline in all_cat_headline]

            cleaned_headline=pd.DataFrame(data=cleaned_headline)

            #Bag of words for particular category
            np.apply_along_axis(self.bagOfWords,1,cleaned_headline,cat_index)



        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):

            #Calculating prior probability p(c) for each class
            prob_classes[cat_index]=np.sum(self.category==cat)/float(self.category.shape[0])

            #Calculating total counts of all the words of each class

            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added

            #get all words of this category
            all_words+=self.bow_dicts[cat_index].keys()


        #combine all words of every category & make them unique to get vocabulary -V- of entire training set

        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]

        #computing denominator value
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])


        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]
        self.cats_info=np.array(self.cats_info)


    def getHeadlineProb(self,test_headline):


        likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each class

        #finding probability w.r.t each class of the given test example
        for cat_index,cat in enumerate(self.classes):

            for test_token in test_headline.split(): #split the test example and get p of each test word


                #get total count of this test token from it's respective training dict to get numerator value
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1


                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])

                #To prevent underflow, log the value
                likelihood_prob[cat_index]+=np.log(test_token_prob)

        post_prob=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])

        return post_prob


    def test(self,test_set):

        predictions=[] #to store prediction of each headline
        for headline in test_set:

            
            cleaned_headline=preprocess_string(headline)

            #get the posterior probability of every headline
            post_prob=self.getHeadlineProb(cleaned_headline) #get prob of this headline for both classes


            predictions.append(self.classes[np.argmax(post_prob)])
        for i in range(len(predictions)):
            if(predictions[i]==1):
                print("Sports")
            elif(predictions[i]==2):
                print("Politics")
            elif(predictions[i]==3):
                print("Entertainment/Television")
            elif(predictions[i]==4):
                print("International News")
            elif(predictions[i]==5):
                print("Technology")
        return np.array(predictions)

training_set=pd.read_csv('news.csv',sep=',') # reading the training data-set

#getting training set headline labels
y_train=training_set['Category'].values
x_train=training_set['Title'].values


from sklearn.model_selection import train_test_split
train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
classes=np.unique(train_labels)

# Training phase....

nb=MultinomialNaiveBayes(classes)
nb.train(train_data,train_labels)

# Testing phase
#pclasses=nb.test(test_data) #testing against huge data set from original dataset
test1=['India lift the world cup after 5 years']
print('\n The Category of: ',test1[0],'...is')
pclasses=nb.test(test1)


test2=['Donald Trump agrees for peace talks with North Korea']
print('\nThe Category of: ',test2[0],'...is')
pclasses=nb.test(test2)

test3=['Elon Musk Just Released a Rap Song About Harambe the Gorilla']
print('\nThe Category of: ',test3[0],'...is')
pclasses=nb.test(test3)

test4=['Googles most secure login system now works on Firefox and Edge, too']
print('\nThe Category of: ',test4[0],'...is')
pclasses=nb.test(test4)

test5=['Trump turns US policy in central america on its head']
print('\nThe Category of: ',test5[0],'...is')
pclasses=nb.test(test5)





'''''
from sklearn.metrics import accuracy_score,classification_report

my_tags=['1','2','3','4','5']

print(classification_report(test_labels, pclasses,target_names=my_tags))


from sklearn.metrics import confusion_matrix
import numpy as np



cm = confusion_matrix(test_labels, pclasses)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
print('Confusion Matrix: \n',cm)
print('Recall: ',np.mean(recall))
print('Precision: ',np.mean(precision))
'''''


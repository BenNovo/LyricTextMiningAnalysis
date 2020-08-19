import csv
import nltk
from statistics import mean 
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np
import pandas as pd
import random
np.random.seed(2018)
import pyLDAvis.gensim
import pickle 
import pyLDAvis
#nltk.download('wordnet')
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

vader_model = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

#Preprocessing helper methods
def lemmatize_stemming(text):
    return lemmatizer.lemmatize(text)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
           # result.append(token)
            result.append(lemmatize_stemming(token))
    return result

def NERPreprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

finalData = {}
#Parse CSV
with open('billboard_lyrics_1964-2015.csv', encoding = 'latin1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = True
    sentimentData = {}
    lyricData = {}
    for row in csv_reader:
        if first:
            first = False
        else:
            lyrics = str(row[4])
            if(len(lyrics) < 3):
                continue
            else:
                sent = vader_model.polarity_scores(row[4])['compound']
                year = str(row[3])
                if(year in sentimentData):
                    sentimentData[year].append(sent)
                else:
                    sentimentData[year] = [sent]
                if(year in lyricData):
                    lyricData[year].append(lyrics)
                else:
                    lyricData[year] = [lyrics]
        
        
     
    yearsToLyricsOfPopTopics = {}   
    #Find annual popular topic and sentiment value
    yearTopics = {}
    for key, value in lyricData.items():
        rand = random.randint(0, len(value) - 1)
        trainingList = []
        for i in range (0, 50):
            try:
                trainingList.append(value.pop(rand))
                rand = random.randint(0, len(value) - 1)
            except Exception as e:
                print(e)
                print(rand)
                print(len(trainingList))
                print(len(value))
        trainingSeries = pd.Series(trainingList)
        processed_docs = trainingSeries.map(preprocess)
        yearTrainingDict = gensim.corpora.Dictionary(processed_docs)
        
        yearTrainingDict.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        bow_corpus = [yearTrainingDict.doc2bow(doc) for doc in processed_docs]
        
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=yearTrainingDict, passes=2, workers=2)
        
        testingList = value
        
        yearTopics[key] = {}
        for item in testingList:
            bow_vector = yearTrainingDict.doc2bow(preprocess(item))
            sortedTopic = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])
            if(int(sortedTopic[0][0]) in yearTopics[key]):
                yearTopics[key][sortedTopic[0][0]] += 1
            else:
                yearTopics[key][sortedTopic[0][0]] = 1
        year = sorted(yearTopics[key].items(), key=lambda item: item[1], reverse = True)
        sentAvg = mean(sentimentData[key])
        finalData[key] = {sentAvg : []}

        for wordid, score in lda_model.get_topic_terms(year[0][0], 20):
            if(yearTrainingDict[wordid] not in finalData[key][sentAvg]):
                finalData[key][sentAvg].append(yearTrainingDict[wordid])
        
        popular = year[0][0]
        for item in testingList:
            bow_vector = yearTrainingDict.doc2bow(preprocess(item))
            sortedTopic = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])
            if(int(sortedTopic[0][0]) == popular):
                if(key in yearsToLyricsOfPopTopics):
                    yearsToLyricsOfPopTopics[key].append(item)
                else:
                    yearsToLyricsOfPopTopics[key] = [item]
        '''if(key == '2015'):
            # Visualize the topics
            LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, bow_corpus, yearTrainingDict)
            pyLDAvis.save_html(LDAvis_prepared, 'LDA_Visualization.html')'''
            
    #Find NER words
    yearsToPopularNERTopics = {}
    for key, value in yearsToLyricsOfPopTopics.items():
        for song in value:
            doc = nlp(song)
            items = [x.text for x in doc if x.pos_ == "NOUN"]
            if key in yearsToPopularNERTopics:
                yearsToPopularNERTopics[key] += (Counter(items).most_common(3))
            else:
                yearsToPopularNERTopics[key] = Counter(items).most_common(3)
                
    for k, v in yearsToPopularNERTopics.items():
        for item in v:
            if(item[0] not in finalData[key][sentAvg]):
                finalData[key][sentAvg].append(item[0])
            
print(finalData)
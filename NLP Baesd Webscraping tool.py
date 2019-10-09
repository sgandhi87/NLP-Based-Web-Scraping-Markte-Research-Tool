#!/usr/bin/env python
# coding: utf-8

# In[49]:


################################CAPSTONE PROJECT - TEXT TO REPORT SUMMARIZER#######################################

################################IMPORT PACKAGES####################################################################
import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load
import requests

import bs4 as bs
import urllib.request
import re
import nltk
import heapq
import json
from nltk import ngrams
import os

from nltk.corpus import stopwords
import numpy as np
import networkx as nx
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from urllib.request import urlopen
import nltk
import spacy

from urllib.request import urlopen
import nltk
from itertools import chain
import re
import nltk
from nltk.corpus import wordnet as wn
#from nltk.util import ngrams


# In[73]:


#######Loading spacy english nlp module##################
spacy_nlp = spacy.load('en_core_web_sm')




###########Extracting text from a web link #########################
def get_text(page):
    url = urllib.request.urlopen(page).read()

    soup = bs.BeautifulSoup(url,'lxml')

    #print(soup.prettify)

    #Converting extracted data to text format

    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.text
    #print(text)

    #Generating Sentences

    text = re.sub(r'\[[0-9]*\]',' ',text)            
    text = re.sub(r'\s+',' ',text)    
    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)
    
    return text


#######################Synonym Function to get the list of similar words#########################
def get_synonyms(word):
    synonyms = []
    for w in word:
        for syn in wn.synsets(w):
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))



##########################Defining Keywords For Each Indicator##################

Investment_Overall_List = ["investment", "merger","acquisition", "money","funding","grants",
                               "seed", "raised", "million", "billion", "dollars", "series", "raised"]
Investment_Funding_List = ["raised", "raises", "Raised", "Raises", "funding"]

Problems_List = ["problem", "problems", "threat", "challenges", "hurdle","drawback", "pain", "negative"]

UseCase_List = ["service", "make", "provide", "solve", "utilization", "product"," users", "technology"]

KeyPlayer_List = ["CEO", "CIO", "CFO", "Co-Founder", "CMO", "CHIEF", "COO", "Partner", "co-founder", "CTO"] 

TargetIndustry_List = ["retail", "healthcare","sector", "manufacturing", "health", "fitness", "welness","music", "banking", 
                      "insurance", "ecommerce","logistics", "agriculture", "hospitality", "robotics", "education",
                       "telecom", "agriculture","mining","utilities","construction","manufacturing","wholesale",
                       "transportation","finance", "estate","education","arts","accommodation",
                       "trade","warehousing","insurance","forestry","fishing","oil","recreation","hunting","forestry",
                       "remediation","entertainment","farming"]

SpecificTech_List = ["Biometrics","3D", "AR", "VR", "Drones", "Cyber", "Artificial", "AI", "ML", "Reinforcement", 
                     "SaaS", "PaaS", "IaaS", "5G","Serverless","Cloud","IOT", "Blockchain", "Robotics",
                     "CX","Cybersecurity", "Voice", "Nanotechnology", "Quantum", "RFID"]

BusinessMetric_List = ["sales", "revenue", "growth","profit","valuation"]





##############Extracting Key Funding Related Words###############################
def get_KeyWords(page, WordList):
    
    sentences = nltk.sent_tokenize(get_text(page))
    stop_words = nltk.corpus.stopwords.words('english')
        
    Wlist1 = []
    post = []
    lst = []
    post1 = []
    list3 = []
    
    for sentence in sentences:
        for w in WordList:
                if w in nltk.word_tokenize(sentence) or nltk.word_tokenize(sentence.lower()) :
                    for w1 in ["series", "seed"]:
                        if w1 in nltk.word_tokenize(sentence.lower()):
                                pre1, term1, post1 = sentence.lower().partition(w1)
                                post1 = post1.split(maxsplit=1)[:1]
                                post1.append(w1.upper())
                                if "$" in nltk.word_tokenize(sentence):
                                    pre, term, post = sentence.partition("$")
                                    pre = pre.rsplit(maxsplit=2)[-2:]
                                    post = post.split(maxsplit=2)[:2]
                                    post[:0] = "$"
                        
                            
                        lst = post1[::-1] + post
                        
                                        

                    
                
                else:
                    pre = []
                    pre1 = []
                    post = []
                    lst = []
                    list3 = []

                Wlist1.append(lst)
                
        k = sorted(Wlist1)
        Wlist3 = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
        list2 = [x for x in Wlist3 if x]
        
                    
    return(list2)            



#########################Removing duplicates from above Keyword list #########################

def remove_duplicate (lst):
    
    list3 = []
    for ls in lst:
        list3.append(" ".join(ls))
    

    final = ', '.join(list3)
    return final    





##############Extracting Key Words for Key Player###############################
def get_KeyWords1(page, WordList):
    
    sentences = nltk.sent_tokenize(get_text(page))
    stop_words = nltk.corpus.stopwords.words('english')
        
    Wlist1 = []
    post3 = []
    list2 = []
    
    for sentence in sentences:
        for w in WordList:
                if w in nltk.word_tokenize(sentence):
#                     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
#                         if (pos == 'NNP'):
#                             post3.append(word)
#                     post3.append(w)
#                     post3[::-1]
                    
                   
                    
                    
                    pre, term, post3 = sentence.partition(w)
                    pre = pre.rsplit(maxsplit=2)[-2:]
                    post3 = post3.split(maxsplit=3)[:3]
                    post3.append(w)
                    post3[::-1]
                    
                else:
                    pre = []
                    post3 = []
                    list2 = []

                Wlist1.append(post3)
                
        k = sorted(Wlist1)
        Wlist3 = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
        list3 = [x for x in Wlist3 if x]

     
    return str(" ".join(chain.from_iterable(list3)))






##############Extracting Key Words for Tech & Industry###############################
def get_KeyWords2(page, WordList):
    
    sentences = nltk.sent_tokenize(get_text(page))
    stop_words = nltk.corpus.stopwords.words('english')
        
    Wlist3 = []
    post3 = []
    lst = []
    
    for sentence in sentences:
        for w in WordList:
                if w in nltk.word_tokenize(sentence):
                    lst.append(w)
                
                elif w in nltk.word_tokenize(sentence.lower()):
                    lst.append(w)
    k = sorted(lst)
    Wlist3 = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
    list3 = [x for x in Wlist3 if x]    
             
    return str(",".join((Wlist3)))




########Further Preprocess########
def preprocess(x):
    x = re.sub("', '"," ", x)
    x = re.sub("’ "," ", x)
    x = x.strip()
    return x





##############Extracting Summary for Use Case & Problems###############################

def get_summary(page, Wordlist):

    text = get_text(page)
    sentences = nltk.sent_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    
    
    #Capturing Product related senstences
    
    sentInfo = []
    for sentence in sentences:
        for w in get_synonyms(Wordlist):
            if w in nltk.word_tokenize(sentence.lower()):
                sentInfo.append(sentence)
    sentInfo = str(sentInfo)
    
                       
        
    #Determine word importance
    sentences2 = nltk.sent_tokenize(sentInfo)
    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)
    
    word2count = {}  
    for word in nltk.word_tokenize(clean_text):     
        if word not in stop_words:                  
            if word not in word2count.keys():
                word2count[word]=1
            else:
                word2count[word]+=1
    for key in word2count.keys():                   
        word2count[key]=word2count[key]/max(word2count.values())

    # Calculate the score

    sent2score = {}
    for sentence in sentences2:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if len(sentence.split(' '))<60:
                    if sentence not in sent2score.keys():
                         sent2score[sentence]=word2count[word]
                    else:
                        sent2score[sentence]+=word2count[word]

    
    best_sentences = heapq.nlargest(2,sent2score,key=sent2score.get)
    
    
    summary =  preprocess(''.join(best_sentences))
    return summary





####Extracting Investment Entity ########
def entity(page, Key):
    
#     financial_list = ['ORG', 'PRODUCT', 'LOC', 'GPE']
    document_I = spacy_nlp(get_summary(page,Investment_Overall_List))
            
    dict_text = {} 
    list2 = []
    
    for element in document_I.ents:
        key = element.label_
        value = element.text
        if key in Key:
            if key not in dict_text.keys():
                dict_text[key] = [value]
            else:
                if value not in dict_text[key]:
                    dict_text[key].append(value)
    list3 = ",".join(chain.from_iterable(list(dict_text.values())))
    return str(list3)



########################Removing Stop Words###################

def remove_stop_words(list1):
    
    stop_words = nltk.corpus.stopwords.words('english')
    
    
       
    for word in list1:     
        if word not in stop_words:
            list1.remove(word)
    return str(list1)        


# In[83]:


#####################Please Press Shift + Enter and Paste the URL link in the box and press Enter again##########


page = input("Please enter a link: ")

Funding = remove_duplicate(get_KeyWords(page,Investment_Funding_List))
Company_Name = entity(page, ["ORG"])
Location = entity(page, ["GPE", "LOC"])
KeyPlayer = get_KeyWords1(page,KeyPlayer_List)
TargetIndustry = get_KeyWords2(page,TargetIndustry_List)
SpecificTech = get_KeyWords2(page,SpecificTech_List)


UseCase_Summary = get_summary(page,UseCase_List)
Problem_Summary = get_summary(page,Problems_List)
Link = str(page)



#########################Extracting Source Name##################################

def source_page(page):
    
    page1 = re.sub(r'[0-9]+', '', page)
    page1 = re.sub('[/,-]', '_', page1)
    page1 = page1.replace('https:','')
    page1 = page1.replace('_+','_')
    page1 = re.sub('_+', '_', page1)
    page1 = page1.strip('_')
    page1_list = page1.split("_")
    page1_list = page1_list[:2]
    page1_list1 = "_".join(page1_list)

    return page1_list1



###############Converting Output Into CSV Format#######################################

col_names =  ['Funding Type ', 'Investors', 'Location', "Problems & Challenges", "Key Employees", "Use Case",
              "Technology", "Target Industry", "Source_URL"]
row_names = ['Outcome']

my_df  = pd.DataFrame(columns = col_names)
my_df.loc[len(my_df)] = [Funding, Company_Name,Location, Problem_Summary, KeyPlayer,UseCase_Summary,
                        SpecificTech,TargetIndustry, Link]

pd.set_option('max_colwidth', 500)
my_df2 = my_df.transpose()

my_df2.columns = ['Outcome']
my_df2.index.name = 'KPI'
my_df2.to_csv("Report2.csv")
os.rename('Report2.csv', 'Report_'+str(source_page(page))+'.csv')
my_df2


# In[ ]:





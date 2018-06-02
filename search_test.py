"""
Referred : http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
http://aakashjapi.com/fuckin-search-engines-how-do-they-work/
"""

import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pickle
from scipy import spatial
import re
import os
import numpy as np
def stem_doc(doc,stemmer): #Not Used
	return [stemmer.stem(word) for word in doc]

def remove_stop(doc):
	#filtered_words = 
	return [word for word in doc if word not in stopwords.words('english')]

#input = ["<Doc_0>","<Doc_1>","<Doc_2>"]
#output = {0:[word1,word2,...],1:[word1,word2,...]}
def process_files(files):
	file_to_terms = {}
	for idx in range(len(files)):
		pattern = re.compile('[\W_]+')
		file_to_terms[idx] = files[idx].lower();
		file_to_terms[idx] = pattern.sub(' ',file_to_terms[idx])
		re.sub(r'[\W_]+','', file_to_terms[idx])
		file_to_terms[idx] = remove_stop(file_to_terms[idx].split())
	return file_to_terms

#input = [word1, word2, ...]
#output = {word1: [pos1, pos2], word2: [pos2, pos434], ...}
def index_one_file(termlist):
	fileIndex = {}
	for index, word in enumerate(termlist):
		if word in fileIndex.keys():
			fileIndex[word].append(index)
		else:
			fileIndex[word] = [index]
	return fileIndex

#input = {filename: [word1, word2, ...], ...}
#res = {filename: {word: [pos1, pos2, ...]}, ...}
def make_indices(termlists):
	total = {}
	for filename in termlists.keys():
		total[filename] = index_one_file(termlists[filename])
	return total

#input = {filename: {word: [pos1, pos2, ...], ... }}
#res = {word: {filename: [pos1, pos2]}, ...}, ...}
def fullIndex(regdex):
	total_index = {}
	for filename in regdex.keys():
		for word in regdex[filename].keys():
			if word in total_index.keys():
				if filename in total_index[word].keys():
					total_index[word][filename].extend(regdex[filename][word][:])
				else:
					total_index[word][filename] = regdex[filename][word]
			else:
				total_index[word] = {filename: regdex[filename][word]}
	return total_index

def extract_data():
	categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
	twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
	#print twenty_train.target_names			
	#print len(twenty_train.data)
	#print("\n".join(twenty_train.data[0].split("\n")[:3]))
	#print(twenty_train.target_names[twenty_train.target[0]])
	#for t in twenty_train.target[:10]:
	#	print(twenty_train.target_names[t])
	return twenty_train

def cosine(vector1,vector2): #Func. not used
	print np.linalg.norm(np.array(vector1)) #Gives mismatch!
	return (1 - spatial.distance.cosine(vector1,vector2)) #SImilarity = 1-dist.

def getIndex(full=False):
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	if 'total_index.p' not in files:
		twenty_train = extract_data()
		if full==False:
			file_to_terms = process_files(twenty_train.data[:100])
		else:
			file_to_terms = process_files(twenty_train.data)
		total = make_indices(file_to_terms)
		total_index = fullIndex(total)
		pickle.dump(total_index,open("total_index.p","wb"))
	else:
		total_index = pickle.load(open("total_index.p","rb"))
	return total_index

def getTfIdfIndex(full=False): #Pickling won't work. So compute always
	twenty_train = extract_data()
	count_vect = CountVectorizer(stop_words=stopwords.words('english'))
	tfidf_transformer = TfidfTransformer()
	if full==False:
		train_data = twenty_train.data[:100]
	else:
		train_data = twenty_train.data
	train_counts = count_vect.fit_transform(train_data)
	tfidf_index = tfidf_transformer.fit_transform(train_counts)
	#print tfidf_index.shape
	query_count = count_vect.transform(['god']) #TRANSFORM NEEDS A LIST
	#print query_count.shape
	pickle.dump(tfidf_index,open("tfidf_index.p","wb"))
	#pickle.dump(count_vect,open("count_vect.p","wb"))
	#pickle.dump(tfidf_transformer,open("tfidf_transformer.p","wb"))
	return (tfidf_index,count_vect,tfidf_transformer)

def one_word_query(word,invertedIdx):
	pattern = re.compile('[\W_]+')
	word = pattern.sub(' ',word)
	if word in invertedIdx.keys():
		return [fN for fN in invertedIdx[word].keys()]
	else:
		return []

total_index=getIndex()
word=raw_input("Please enter the query: ")
doc_idxs = one_word_query(word,total_index)
print ("Unranked: ",doc_idxs)
tfidf_index,count_vect,tfidf_transformer = getTfIdfIndex()

query_count = count_vect.transform([word])
#print query_count.shape
query_tfidf = tfidf_transformer.transform(query_count)
#print "Here: ",query_tfidf.shape
#print tfidf_index.shape
#print tfidf_index[0].shape
similarities=[]
for idx in doc_idxs:
	similarities.append(cosine_similarity(tfidf_index[idx],query_tfidf)[0][0])

#print similarities
print ("Ranked order: ",[x for _,x in sorted(zip(similarities,doc_idxs))])

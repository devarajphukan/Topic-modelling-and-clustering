import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, gutenberg
from gensim import corpora, models
import gensim
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer

def stem_tokens(tokens):
	stemmed = []
	for item in tokens:
		stemmed.append(PorterStemmer().stem(item))
	return stemmed

def pos_tag(text):

	tokens = nltk.word_tokenize(text)
	# tokens = stem_tokens(tokens, stemmer)
	tokens = nltk.pos_tag(tokens)
	return tokens

def clean_text(corpus):

	clean_corpus = []
	for text in corpus :
		text = str(filter(lambda x:ord(x)>31 and ord(x)<128,text)).lower().strip()
		text = re.sub("[^a-zA-Z]", " ", text) # Removing numbers and punctuation
		text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text) # Removing very long words above 10 characters
		text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b"," ",text) # Removing single characters (e.g k, K)
		text = re.sub(" +"," ", text) # Removing extra white space
		clean_corpus.append(text.strip())
	return clean_corpus

def get_n_grams(corpus):

	n_gram_corpus = []

	for temp_doc in corpus :
		
		bigrams = zip(temp_doc.split(" "), temp_doc.split(" ")[1:])
		bigrams = ["_".join(x) for x in bigrams]
		
		trigrams = zip(temp_doc.split(" "), temp_doc.split(" ")[1:], temp_doc.split(" ")[2:])
		trigrams = ["_".join(x) for x in trigrams]
		
		quadrigrams = zip(temp_doc.split(" "), temp_doc.split(" ")[1:], temp_doc.split(" ")[2:], temp_doc.split(" ")[3:])
		quadrigrams = ["_".join(x) for x in quadrigrams]

		n_gram_corpus.append(tuple(bigrams,trigrams,quadrigrams))
	
	return n_gram_corpus

def get_stop_words() :

	stopset = stopwords.words('english')
	freq_words = ['nd','st']
	for i in freq_words :
		stopset.append(i)
	return stopset

def unigram_filtering(corpus) :

	unigrams = []
	for temp_doc in corpus :
		temp_doc = pos_tag(temp_doc)
		temp_li = []
		for word in range(len(temp_doc)) :
			if (temp_doc[word][0] not in stopset and temp_doc[word][0] in topTfidfWords) :# and (temp_doc[word][1] == 'NN' or temp_doc[word][1] == 'JJ') :
				temp_li.append(temp_doc[word][0])
		unigrams.append(temp_li)
	return unigrams

def get_top_tfidf_words(corpus, num_top_words) :
	
	vectorizer = TfidfVectorizer(analyzer= 'word', stop_words = stopset, lowercase = True)
	tfidf_features = vectorizer.fit_transform(corpus)
	indices = np.argsort(vectorizer.idf_)[::-1]
	features = vectorizer.get_feature_names()
	top_features = [features[i] for i in indices[:num_top_words]]
	return top_features

def lda_model(text_corpus, n_topics, n_words, n_passes) :
	
	dictionary = corpora.Dictionary(text_corpus)
	corpus = [dictionary.doc2bow(text) for text in text_corpus]
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = n_topics, id2word = dictionary, passes = n_passes)
	for topics in ldamodel.print_topics(num_topics = n_topics, num_words = n_words) :
		print topics,"\n"

if __name__ == "__main__" :

	with open('science.txt', 'r') as content_file:
		text_content = content_file.read().split("\n")

	clean_content = clean_text(text_content[:1000])
	
	stopset = get_stop_words()
	
	topTfidfWords = get_top_tfidf_words(clean_content,1800)
	final_corpus = unigram_filtering(clean_content)

	num_topics = 8
	num_words = 10
	num_passes = 200

	lda_model(final_corpus, num_topics, num_words, num_passes)


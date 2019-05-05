from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import json

'''
Once completed running this script will generate and save the word vectors 
in a pickle file.
'''

'''
You want texts to be a 2 dimensional list.  Each list should be a sentence.
Each sentence should be represented as a list of words.  The more sentences
in texts the better and broader word vectors you will get.

Some functions that will be useful:
open('FILEPATH','r') Returns the file stored at the relative 'FILEPATH'
					 in a format that can be read.
json.load(file)      Returns the parsed contents of a file.
sent_tokenize(text)  Accepts a string that contains some number of sentences.  
					 Returns a list where each element is a sentence from 
					 text in order.
word_tokenize(text)  Accepts a string that contains some number of words.
					 Returns a list where each element is a word from the
					 text in order.
sum(listOfLists, []) Concatenates and returns lists in listsOfLists.

Feel free to mess around with these functions in the terminal to understand
better what they do.'''

#CODE HERE

texts = 

'''
The dimensionality of your word vectors.  This is more of a judgement call.
You want to make it large enough where it's sufficiently expressive and small
enough where it's not excessive.  Remember the idea of wordVectors is to build
a rich numerical representation of words.
'''
wordVecDim =

'''
Word2Vec is a popular algorithm that learns word vectors based on contextual
relationships.  Feel free to read more online about how the model works.  But
all you need to know for this project is that they will allow you to represent
a word as a numerical vector of dimension 'size'.
'''
wordVectors = Word2Vec(sentences=sum(texts,[]), size=wordVecDim).wv

#Save the word vectors and dimensionality to be used later
pickle.dump((wordVectors, wordVecDim),open('wv.pkl','wb'))

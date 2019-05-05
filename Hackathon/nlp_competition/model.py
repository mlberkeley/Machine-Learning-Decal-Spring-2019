import tensorflow as tf
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize

'''
Setting MODE to 0 trains the model
Setting MODE to 1 allows the model to run on user input 
Setting MODE to 2 creates final 'predictions.json' file
'''
MODE = 0
'''
Lets you toggle whether you want to used previously saved
weights from the last run or override any previously saved 
weights.
'''
USE_OLD = True

print("Loading word vectors")
#Load wordVectors and wordVecDim from pickle file
wordVectors, wordVecDim = pickle.load(open('wv.pkl','rb'))

'''
Should return the word vector of a word.  If the word is in wordVectors
return the appropriate word vector.  If not return a word vector of the
same dimension (wordVecDim) but filled with zeros. (Hint: np.zeros)
wordVectors[word] will return a wordVecDim dimensional vector
if word is in the set of keys in wordVectors.  You can check
if word is in the set of keys in wordVectors using 
'word in wordVectors'.
'''
def getVector(w):

	#CODE HERE


'''
Given a string containing words return a list of dimensionality
equal to the number of words in 's'.  Each element of the list
should be the appropriate word vector for the corresponding word
in 's'. Simply put return a list of word vectors of 's'.
Remember we can use word_tokenize(string) to partition a given
string into a list of words.  Also consider the function you
wrote just above.
'''
def wordVectorize(s):

	#CODE HERE

#MODE=0 we need training data, MODE=1 we need no data, MODE=2, we need test data
if(MODE==0 or MODE==2):
	print("Loading data...")

	#Load the training data if training
	if(MODE==0):
		labeledReviews = json.load(open('trainingDataset.json','r'))

	#Load the testing data if testing
	if(MODE==2):
		testReviews = json.load(open('testDataset.json','r'))

	'''
	extract accepts reviews: list of dictionaries representing reviews.
	The dictionaries must hold keys 'title' and 'review_text'.  If ratingsPresent
	the 'rating' key should also exist
	'''
	def extract(reviews, ratingsPresent):

		#Expression that evaluates to to title of r
		titles = [_______ for r in reviews]
		#Expression that evaluates to to review text of r
		texts = [_______ for r in reviews]
		if(ratingsPresent):
			'''
			Expression that evaluates to 1/True if 
			review is positive and 0/False if review is negative.
			'''
			ratings = [_______ for r in reviews]
		#Expression that evaluates to total number of reviews
		numReviews = ______

		#Expression that evaluates to a list of the word vectors of t
		titleVectors = [_______ for t in titles]
		#Expression that evaluates to a list of the word vectors of t
		textVectors = [_______ for t in texts]

		'''
		Should return true if and only if both titleVectors[i] and 
		textVectors[i] have positive length.  That is a non zero
		number of words were found in either.  This is to avoid 
		null input errors later.
		'''
		def validReview(i):
			#CODE HERE


		'''
		Given a list 'l' where l[i] corresponds to some attribute of
		the ith review, return a list containing all elements of l in
		order where the ith review is valid.  Being valid is as defined
		in the above function
		'''
		def onlyValid(l):
			#CODE HERE


		#Return the appropriate values depending on ratingsPresent
		if(ratingsPresent):
			ratings, titleVectors, textVectors = onlyValid(ratings), onlyValid(titleVectors), onlyValid(textVectors)
			return ratings, titleVectors, textVectors
		else:
			titleVectors, textVectors = onlyValid(titleVectors), onlyValid(textVectors)
			return titleVectors, textVectors

	#Data preparation for training
	if(MODE==0):
		'''
		Should be some value between 0 and 1.  Suppose you used 0.4.
		Then 40% of the labeled data would be used for training and
		60% would be used for validation.  Note that we generally want
		to use the majority of the data for training, but there is no
		magic number.
		'''
		tvCutoff = int(len(labeledReviews)*_____)
		trReviews, valReviews = labeledReviews[:tvCutoff], labeledReviews[tvCutoff:]

		'''
		Given a list of reviews return 2 lists. The first contains all
		positive reviews and the latter contains all negative reviews.
		'''
		def sentimentSplit(reviews):
			ratings, titles, texts = extract(reviews, ratingsPresent=True)
			numValidReviews = len(titles)

			#Expression that evalutes to true if a rating is positive
			positiveReviews = [(titles[i], texts[i], ratings[i]) for i in range(numValidReviews) if _______]

			#Expression that evalutes to true if a rating is negative
			negativeReviews = [(titles[i], texts[i], ratings[i]) for i in range(numValidReviews) if _______]
			
			return positiveReviews, negativeReviews

		positiveTrain, negativeTrain = sentimentSplit(trReviews)
		numPositive = len(positiveTrain)
		numNegative = len(negativeTrain)

		valRatings, valTitles, valTexts = extract(valReviews, ratingsPresent=True)
		numVal = len(valTitles)

		valSet = [(valTitles[i], valTexts[i], valRatings[i]) for i in range(numVal)]

	#Data preparation for testing
	if(MODE==2):
		teTitles, teTexts = extract(testReviews, ratingsPresent=False)
		numTest = len(teTitles)

		testSet = [(teTitles[i], teTexts[i]) for i in range(numTest)]

print("Setting up graph...")

'''
titlePlaceholder will contain the numerical representation of the title
of a single review in your computational graph.

Search tf.placeholder for how to create a tensorflow placeholder.

Use tf.float32 as your datatype.
Be careful when specifying the shape attribute.  Remember that
titles can be of variable length.  'None' is used to represent
a wild card dimension.  For example both [4,2,5] and [4,6,5] would
be valid inputs to a placeholder of shape [4,None,5]
Name is optional
'''
titlePlaceholder = ______
'''
Same as above but for a single review text as opposed to a single
title.
'''
textPlaceholder = _______
'''
Placeholder for a single float value.  This value should be 1 when
the given review is positive and 0 when not.  Note this value is not
used in making a prediction only for computing loss while training.
'''
y = ______

'''
Dimensionality of the cell/hidden state of your LSTM.
Too small and your model won't be sufficiently expressive.
Too large and your model will quickly overfit.
'''
cellSize = ____

'''
Define your LSTM cell for the title LSTM.

Search tf.nn.rnn_cell.LSTMCell to find out how to make one

You should only have to supply one parameter, but feel free
to modify any other parameters you understand.
'''
titleLSTM = tf.nn.rnn_cell.LSTMCell(________)

'''
Define your LSTM that will process the information
contained in titlePlaceholder (a single review title)

Search tf.nn.dynamic_rnn to find out how to make one.

You only have to supply cell, inputs, dtype, and scope.
It's up to you to figure out what goes in cell and inputs.
Feel free to use tf.float32 as the datatype again.

The reason we must specify scope is that the scope name
is used to save the parameters of the LSTM.  Since we
are going to have another LSTM for the review texts we
need to specify different scopes for the two of them so
their weights don't collide in storage.

a, b are intentionally left ambiguous.  You will have
to read in the documentation what they are and figure
out what you need.  If you find you only need one of them
simply replacing 'a, b' with either '_, b' or 'a, _' will
be more efficient as you don't save the unecessary information.
'''
a, b = tf.nn.dynamic_rnn(______)


'''
Using a and b (or one of the two) determine the final cell
state of the LSTM.  We then reshape it into a column vector
of dimensionality cellSize
'''
titleCellState = tf.reshape(________,[cellSize, 1])

'''
Repeat the above steps for the textLSTM.
Define your LSTM that will process the information
contained in textlaceholder (a single review text)
'''
textLSTM = ________

a, b = tf.nn.dynamic_rnn(_____)

textCellState = tf.reshape(______,[cellSize, 1])

'''
Concatenate the two cell states into a single vector of
dimensionality [2*cellSize, 1]

Search tf.concat for how to do this.
'''
combinedState = tf.concat(________)

'''
Returns a tensor variable of the specified size that is
initialized using a truncated normal distribution of stddev 0.1.
'''
def weight(s):
	return tf.Variable(tf.truncated_normal(s, stddev=0.1))

#Dimensionality of the hiddenLayer.  hiddenLayer should be [hiddenSize, 1]
hiddenSize = _____

'''
Size should allign with the expected input and desired output
Keep in mind a m x n matrix times a n x 1 vector results in a
m x 1 vector.
'''
hiddenMap = weight(________)
'''
Define a variable initialized to a zero vector that can be
added as shown below.

Either search tf.Variable or see above for how you can use it.
Hint: tf.zeros
'''
hiddenBias = tf.Variable(______)
'''
Find the tf function that performs the standard relu operation.
Feel to free to use any other nonlinearity if you wish.
'''
hiddenLayer = ______(tf.matmul(hiddenMap, combinedState) + hiddenBias)

'''
predictionMap should be a weight with same dimensionality as
the hidden layer.
'''
predictionMap = weight(______)
'''
The first blank should be filled with the tf function that
accepts a tensor and reduces it to a single value, it's sum.

The second blank should be filled with the tf function that
computes the element wise product of the two given vectors.

Note we are effectively computing the dot product here.
'''
logit = ______(______(predictionMap, hiddenLayer))
'''
Find a tf function that accepts any real number and outputs
a number between 0 and 1.  We're looking for a specific activation
function that allows our output to resemble our target values.
'''
prediction = _____(logit)

'''
Here we are computing the cross entropy loss.  epsilon is simply
a small number that we add in our logarithms to avoid exploding losses.
This is because log(tiny number) diverges to -infinity.

Ignoring epsilon we want our loss to be as follows
If y = 0.  Then the loss should be 0 when prediction = 0 and increase
as prediction increases to 1.
If y = 1.  Then the loss should be 0 when prediction = 1 and increase
as prediction decreases to 0.

Note the negative sign outside the whole expression.
'''
epsilon = 1e-2
loss = -(____*tf.log(_____ + epsilon) + ____*tf.log(______ + epsilon))
'''
correct should be True/1 iff
y = 0 and prediction is closer to 0 than 1
or y = 1 and prediction is closer to 1 than 0
Otherwise it should be False/0.
'''
correct = _____

#Learning rate for our model, feel free to adjust between runs
LEARNING_RATE = _____
'''
Find a tf function to minimize your loss.
Feel free to use AdamOptimizer.
'''
trainStep = _____(LEARNING_RATE).minimize(loss)

#Tensorflow session
sess = tf.InteractiveSession()

#For saving/restoring previous weights
saver = tf.train.Saver()

#Restore weights or initialize to new
if(USE_OLD):
	print("Restoring weights...")
	saver.restore(sess,"saved/weights.ckpt")
else:
	sess.run(tf.global_variables_initializer())

'''
No need to modify this.  Computes and prints accuracies.
Positive accuracy is defined as the accuracy of the model
on reviews that were actually positive and vice versa.
'''
def printPerformance(predictions):
	correctNegative = predictions[0][1]/(sum(predictions[0]))
	correctPositive = predictions[1][1]/(sum(predictions[1]))
	normalizedAccuracy = 0.5*(correctPositive+correctNegative)
	print(f"Accuracies")
	print(f"Positive Accuracy: {correctPositive} Negative Accuracy: {correctNegative}")
	print(f"Overall Accuracy: {normalizedAccuracy}")

#TRAIN
if(MODE==0):
	print("Training model...")

	#Number of batches you want to train
	BATCHES = ____
	#Number of reviews you want to learn from in a batch
	BATCH_SIZE = ____

	'''
	We're interested in normalized accuracy, that means
	the average between positive and negative accuracy.
	However our training dataset is skewed with about 4 
	times as many positive as negative reviews.

	For this reason directly sampling from the training
	dataset with no modification to the weight update
	will result in a model that is significantly biased
	towards making positive predictions.

	There are a few approaches to adressing this problem.
	Perhaps the most straightforward is to merly ensure
	that during training the model is equally exposed to
	positive and negative reviews.

	Implementation of this is up to you but make sure
	that you are using the dataset to its fullest extent.
	That is don't simply discard some data.

	Remember positiveTrain holds numPositive positive
	reviews and negativeTrain holds numNegative negative
	reviews.

	Here you may wish to initialize some variables to
	help accomplish this.
	'''

	#CODE HERE

	#loop through batches
	for i in range(BATCHES):
		#Cumulative loss for this batch
		batchLoss = 0
		#List to remember predictions for this batch
		#[[False Positives, True Negatives], [False Negatives, True Positives]]
		predictions = [[0,0],[0,0]]
		for r in range(BATCH_SIZE):
			print(f"Batch {i+1} training {r+1}/{BATCH_SIZE}", end='\r')
			'''
			Depending on r, set the values of the variables 'title', 'text',
			and 'rating'.  This will be used as input to our model.

			How you set these values will depend on how you wish the implement
			the earlier mentioned task.

			You may want to reference/update some variables you initialized
			in the previous 'CODE HERE' block.
			'''
			
			#CODE HERE

			#Train the model and compute the loss and whether or not the model was correct
			#Fill in the appropriate placeholder values
			l, c, _ = sess.run([loss, correct, trainStep], 
							feed_dict={titlePlaceholder:title, textPlaceholder:text, y:rating})
			#Increment the predictions and batchLoss as appropriate.
			predictions[rating][c]+=1;
			batchLoss+=l
		print()
		print(f"Batch {i+1} Loss: {batchLoss/BATCH_SIZE}")
		printPerformance(predictions)
		print("---------")
		'''
		Every SAVE_EVERY batches we compute the performance on the
		validation dataset and save the model weights.  Keep in mind
		the purpose of the validation dataset.
		'''
		SAVE_EVERY = ____
		if(i%SAVE_EVERY==SAVE_EVERY-1):
			print("Validation Performance")
			#Similar tracking to actual training
			valLoss = 0
			predictions = [[0,0],[0,0]]
			for v in valSet:
				#Get necessary information for a single validation review
				title, text, rating = v
				#Fill in the appropriate placeholder values
				l, c = sess.run([loss, correct], 
							feed_dict={titlePlaceholder:___, textPlaceholder:___, y:___})
				predictions[rating][c]+=1;
				valLoss+=l
			print(f"LOSS: {valLoss/numVal}")
			print("Saving weights...")
			printPerformance(predictions)
			saver.save(sess, "saved/weights.ckpt")
		print("---------")

#TEST with user input
if(MODE==1):
	while(True):
		reviewTitle = input("Supply a Review Title: ")
		reviewText = input("Supply text for your Review: ")
		'''
		Fill in appropriate place holder values.  You might
		have to use an earlier function we wrote to parse
		the user input.
		'''
		p = sess.run(prediction, feed_dict={titlePlaceholder:______, 
											textPlaceholder:______})
		if(p > 0.5):
			print(f"POSITIVE with {(p*100):.2f}% confidence.")
		else:
			print(f"NEGATIVE with {((1-p)*100):.2f}% confidence.")
		print()

if(MODE==2):
	print("Making test predictions...")
	predictions = []
	for i in range(numTest):
		print(f"Predicting {i+1}/{numTest}",end='\r')
		title, text = testSet[i]
		#fill in appropriate placeholder values
		pred = sess.run(prediction, feed_dict={titlePlaceholder:____, textPlaceholder:___})
		#Append binary model prediction to list
		predictions.append(float(pred)>0.5)
	'''
	Save model prediction to appropriate json file
	Remember to write your name where requested.
	Just first name is fine.
	'''
	json.dump(predictions, open('YOUR_NAME.json','w'))




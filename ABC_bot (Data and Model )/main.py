# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy
import tflearn
import tensorflow
import random
import json
# the pickle is the process of converting a Python object into a byte stream to store it in a file/database
import pickle
# import our chat-bot intents file
with open("intents.json") as file:
    data = json.load(file)
# we try to ipen some saved data
# try to load in some pickled and that data is gonna be our words,labels, training, output
# so if we have this data already saved we have use it and processed it before
# so we don't have to do what is in except again
# rb means read bytes that is mean we are gonna save our data as bytes
try:
    with open("data.pickle", "rb") as f:
        # we will save this four variables into a pickle file
        words, labels, training, output = pickle.load(f)
# if what's in the try dosen't work successfully we will run the code in except
except:
    words = []
    #classes ,tags
    labels = []
    #  we store different pattern in docs_x
    docs_x = []
#docs_y stands for what intents it's apart of
    docs_y = []
#so that each entry in doc y correspanding to an entry of docs x (maping the pattern to his own entent)
# this is an important step to classify each pattern
# loop through each sentence in our intents patterns
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
# tokenize each word in the sentence
            wrds = nltk.word_tokenize(pattern)
#rather than go through it and append each one we can just extend the list which mean we are going to add all of those words in it
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
 # add to our classes list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
#w.lower convert all our words  into lowercase
# list comprehension
# stem and lower each word and remove duplicates 
#we only stem each word in words list
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
# set it take all words and make sure that there is no duplication
#list convert it into a list and then sorted will sort it
    words = sorted(list(set(words)))

    labels = sorted(labels)
# create our training data and testing
#training list have bunch of bag of words list of 0's and 1's
    training = []
# list of 0's and 1's 
    output = []
# each two are in hot encoded
#neural network only understand numbers so we will create a bag of words that represents all the words of any given pattern
#for output we have 38*16 array if the pattern exist in that tag we put one if don't it put 0
# create an empty array for our output
# for _ means give special meanings and functions to name of vartiables or functions. and to separate the digits of number literal value.
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
# we didn't stem them when we added it to docs x so we stem each word in  docs x (pattern )
        wrds = [stemmer.stem(w.lower()) for w in doc]
    # create our bag of words array
#we will go through all of the different words that is in our document or in the word list now that is stemmed
# the first four we will loop in the sorted list words that we sorted it in line 47
        for w in words:
# we make sure if the w that loop through words in our docs that store the pattern or not
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
# now we are gonna through for our label list we are gonna see where the tag in this list and we are goona set this value to one in our output row
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
# then i turn the two lists into numpy array
    training = numpy.array(training)
    output = numpy.array(output)
# after the code in except run we will save it into a pickle file so we can reuse it withouy running all the abive code
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
# now  we will build our model with tf learn
# we make this step to get ride of previous setting and to reset underlying graph data
tensorflow.reset_default_graph()
# in this step what is gonna do is definnig our input shape that we're expecting in our model
# we are gotting the length of tarinning zero because each trainning in is gonna be the same length
# in the following code we set our input data
net = tflearn.input_data(shape=[None, len(training[0])])
#the next step mean we add this fully connected layer to our neural network
#we have eight neurons for that hidden layers
# the two net variable means that we have two hidden layers  each layer has 8 neurons
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
#now we will  make another layer that allow us to get propabilities for each output
# softmax is gonna through all the output and give us a probability for each neurons in this layer and will be our output for this network
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
# Start training our model
#number of epoch is the amount of time that it is gonna see the same data

model.fit(training, output, n_epoch=2500, batch_size=20, show_metric=True)
model.save("model.tflearn")

# we don't have to do all of these code every time that we want to use the model
# we produce a bag-of-words from user input. This is the same technique as we used earlier to create our training documents.
def bag_of_words(s, words):
    # we create a blank bag of words
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
# the result gives me a list of prediction for each tag
        results = model.predict([bag_of_words(inp, words)])[0]
# the second result will give us the highest probability in our prediction list
        results_index = numpy.argmax(results)
# we will use that index to figure out which respons to actually display
        tag = labels[results_index]
# this condition tell us if the probability is higher than 70 confidence we pock that response
        if results[results_index] >0.5 :
# we will loop into our json file to figure out which responses will figure out the tag
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
    
            print(random.choice(responses))
        else:
            print("It's seem a big problem can't help in it so visit our center ^^")

chat()

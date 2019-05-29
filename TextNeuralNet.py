#! /E:/Anaconda3 python

#Code adpated from Denny Britz's guide in CNN for Text Classification via TensorFlow
#https://github.com/dennybritz/cnn-text-classification-tf

#Imports
import tensorflow as tensorflow
import numpy as numpy

class TextNeuralNet(object):
    """
    Neural Network Object,
    Layers are defined here,
    As well as Accuracy and Loss Scores
    """
    def __init__(
        #self
        self,
        #Length of Sentences
        sequenceLength,
        #Number of Classes in Output Layer (=2, pos and neg)
        numClasses,
        #Size of vocablury, used for defining size of embedding layer
        vocabSize,
        #Dimensions of embeds
        embedSize, 
        #Number of words convo filters will cover
        #eg [3,4,5] means filters that cover 3, 4 and 5 words and total of 3 * num_filters
        filterSizes,
        #Number of filters per filter size
        numFilters
        ):

        #placeholder values
        self.inputX = tensorflow.placeholder(tensorflow.int32, [None, sequenceLength], name = "inputX")
        self.inputY = tensorflow.placeholder(tensorflow.int32, [None, numClasses], name = "inputY")
        self.rate = tensorflow.placeholder(tensorflow.float32, name = "rate")

        #embed layer
        #/cpu:0 forces running on CPU, embed operation currently not GPU supported
        #running on GPU could slow down the process in general due to this
        with tensorflow.device("/cpu:0"), tensorflow.name_scope("embedding"):
            self.W = tensorflow.Variable(
                tensorflow.random_uniform([vocabSize, embedSize], -1.0, 1.0), name = "w")
            self.embedChars = tensorflow.nn.embedding_lookup(self.W, self.inputX)
            self.embedCharsExpanded = tensorflow.expand_dims(self.embedChars, -1)


        poolOuput = []
        for i, filterSize in enumerate(filterSizes):
            with tensorflow.name_scope("conv-maxpool-%s" % filterSize):
                #Convolution Layer
                filterShape = [filterSize, embedSize, 1, numFilters]
                W = tensorflow.Variable(tensorflow.truncated_normal(filterShape, stddev = 0.1), name = "W")
                b = tensorflow.Variable(tensorflow.constant(0.1, shape = [numFilters]), name = "b")
                conv = tensorflow.nn.conv2d(
                    self.embedCharsExpanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv"
                )
                #Create Max Pooling Layer
                h = tensorflow.nn.relu(tensorflow.nn.bias_add(conv, b), name = "relu")
                pooled = tensorflow.nn.max_pool(
                    h,
                    ksize = [1, sequenceLength - filterSize + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "pool"
                )
                poolOuput.append(pooled)
    
        #Combine Pooled Features
        numFiltersTotal = numFilters * len(filterSizes)
        self.hPool = tensorflow.concat(poolOuput, 3)
        self.hPoolFlat = tensorflow.reshape(self.hPool, [-1, numFiltersTotal])

        #Dropout
        with tensorflow.name_scope("rate"):
            self.hDrop = tensorflow.nn.dropout(self.hPoolFlat, rate = self.rate)

        #Score and Prediction
        with tensorflow.name_scope(name = "output"):
            W = tensorflow.Variable(tensorflow.truncated_normal([numFiltersTotal, numClasses], stddev = 0.1), name = "W")
            b = tensorflow.Variable(tensorflow.constant(0.1, shape = [numClasses]), name = "b")
            self.scores = tensorflow.nn.xw_plus_b(self.hDrop, W, b, name = "scores")
            self.predictions = tensorflow.argmax(self.scores, 1, name = "predictions")

        #Calculate Losses
        with tensorflow.name_scope("loss"):
            losses = tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits = self.scores, labels = self.inputY)
            self.loss = tensorflow.reduce_mean(losses)

        #Calculate Accuracy
        with tensorflow.name_scope("accuracy"):
            correctPredicts = tensorflow.equal(self.predictions, tensorflow.argmax(self.inputY, 1))
            self.accuracy = tensorflow.reduce_mean(tensorflow.cast(correctPredicts, "float"), name = "accuracy")
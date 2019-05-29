#! /E:/Anaconda3 python

#Code adpated from Denny Britz's guide in CNN for Text Classification via TensorFlow
#https://github.com/dennybritz/cnn-text-classification-tf

#imports
import numpy
import re

def cleanString(string):
    """
    Cleans data sets by removing noisy data,
    Also ensures that specific punctuation isn't mistaken as part of a word 
    by adding whitespace around it
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def loadDataAndLabels(positiveData, negativeData):
    """
    Loads polarity data from file, splits each text into separate words and generates labels,
    Returns the labelled and split sentences
    """
    # Load positive data...
    positiveExamples = list(open(positiveData, "r").readlines())
    positiveExamples = [s.strip() for s in positiveExamples]
    #Load Negative Data...
    negativeExamples = list(open(negativeData, "r").readlines())
    negativeExamples = [s.strip() for s in negativeExamples]
    # Split by words
    xText = positiveExamples + negativeExamples
    xText = [cleanString(sent) for sent in xText]
    # Generate labels
    positive_labels = [[0, 1] for _ in positiveExamples]
    negative_labels = [[1, 0] for _ in negativeExamples]
    y = numpy.concatenate([positive_labels, negative_labels], 0)
    return [xText, y]
    

def batchIterate(data, batch_size, num_epochs, shuffle=True):
    """
    Creates a batch to be iterated through from the data set
    """
    data = numpy.array(data)
    dataSize = len(data)
    batchesPerEpoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = numpy.random.permutation(numpy.arange(dataSize))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(batchesPerEpoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, dataSize)
            yield shuffled_data[start_index:end_index]


#! /E:/Anaconda3 python

#Code adpated from Denny Britz's guide in CNN for Text Classification via TensorFlow
#https://github.com/dennybritz/cnn-text-classification-tf

#Imports
import tensorflow
import numpy
import os
import time
import datetime
import dataHelper
from TextNeuralNet import TextNeuralNet
from tensorflow.contrib import learn
import csv

#Remove CPU Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-=-=-=-=-=PARAM DEFINITIONS=-=-=-=-=-#
#Use these to quickly change any relevant parameters to adjust the algo

# Data Parameters
tensorflow.flags.DEFINE_string("positive_data_file", "./data/polarityData/eval.pos", "Positive data source TEST SOURCE")
tensorflow.flags.DEFINE_string("negative_data_file", "./data/polarityData/eval.neg", "Negative data source TEST SOURCE")

# Eval Parameters
tensorflow.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tensorflow.flags.DEFINE_string("checkpoint_dir", "E:/Robert Lyons/Robert Lyons/GitHub/SA-Project/runs/1556277843/checkpoint", "Checkpoint directory from training")

# Misc Parameters
tensorflow.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tensorflow.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#assign tensorflow.flags.FLAGS to simply FLAGS
FLAGS = tensorflow.flags.FLAGS

#-=-=-=-=-=END OF DEFINITIONS=-=-=-=-=-#

#Load Data
#Positive and negative data is still split so that accuracy can be evaluated
xRaw, yTest = dataHelper.loadDataAndLabels(FLAGS.positive_data_file, FLAGS.negative_data_file)
yTest = numpy.argmax(yTest, axis = 1)

#Map Data into vocab library
vocabPath = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocabProcessor = learn.preprocessing.VocabularyProcessor.restore(vocabPath)
xTest = numpy.array(list(vocabProcessor.transform(xRaw)))

print("Evaluating Data...\n")

#-=-=-=-=-=EVALUATION STAGE=-=-=-=-=-#
checkpointFile = tensorflow.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tensorflow.Graph()
with graph.as_default():
    sessionConfig = tensorflow.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement
    )
    session = tensorflow.Session(config = sessionConfig)
    with session.as_default():
        #Load saved graph, restore variables
        saver = tensorflow.train.import_meta_graph("{}.meta".format(checkpointFile))
        saver.restore(session, checkpointFile)

        #Get placeholders from Graph
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        #inputY = graph.get_operation_by_name("inputY").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        #Tensors to evaluate
        predicts = graph.get_operation_by_name("output/predictions").outputs[0]

        #Generate Batches
        batches = dataHelper.batchIterate(list(xTest), FLAGS.batch_size, 1, shuffle = False)

        #collect predictions using this
        allPredicts = []

        #dropoutKeepProb at 1.0 so neural model doesn't learn from testing data set
        for xTestBatch in batches:
            batchPredicts = session.run(predicts, {inputX: xTestBatch, dropoutKeepProb: 1.0})
            allPredicts = numpy.concatenate([allPredicts, batchPredicts])

#print accuracy score if y is defined
if yTest is not None:
    #sum total correct
    sumCorrect = float(sum(allPredicts == yTest))

    #sum true positives and negatives
    truePositives = numpy.sum(numpy.logical_and(allPredicts == 1, yTest == 1))
    trueNegatives = numpy.sum(numpy.logical_and(allPredicts == 0, yTest == 0))
    
    #sum false pasitive and negatives
    falsePositives = numpy.sum(numpy.logical_and(allPredicts == 1, yTest == 0))
    falseNegatives = numpy.sum(numpy.logical_and(allPredicts == 0, yTest == 1))
    
    #print all
    print("Total Test Examples: {}".format(len(yTest)))
    print("Accuracy: {:g}".format(sumCorrect/float(len(yTest))))

    print("True Positives: {}".format(truePositives))
    print("True Negatives: {}".format(trueNegatives))
    
    print("False Positives: {}".format(falsePositives))
    print("False Negatives: {}".format(falseNegatives))

#Change numerical eval value to string for readability
readablePredicts = []
for predicts in allPredicts:
    if predicts == 0:
        readablePredicts.append("Negative")
    if predicts == 1:
        readablePredicts.append("Positive")

#Change numerical eval value to string for readability
readableActual = []
for values in yTest:
    if values == 0:
        readableActual.append("Negative")
    if values == 1:
        readableActual.append("Positive")

#save evals to CSV
predictsHeaders = numpy.column_stack(("Tweet", "Predicted Sentiment", "Actual Sentiment"))
predictsReadable = numpy.column_stack((numpy.array(xRaw), readablePredicts, readableActual))
outputPath = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.csv")
print("Saving Evaluation Results To {0}".format(outputPath))
with open(outputPath, "w+") as file:
    csv.writer(file).writerows(predictsHeaders)
    csv.writer(file).writerows(predictsReadable)
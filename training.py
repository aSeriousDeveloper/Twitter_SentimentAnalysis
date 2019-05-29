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

#Remove CPU Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-=-=-=-=-=PARAM DEFINITIONS=-=-=-=-=-#
#Use these to quickly change any relevant parameters to adjust the algo

#Data Loading params
tensorflow.flags.DEFINE_float("dev_sample_percentage", 0.5, "Percentage of training data to use for testing")
tensorflow.flags.DEFINE_string("positive_data_file", "./data/polarityData/tweets.pos", "Positive data source TRAINING SOURCE")
tensorflow.flags.DEFINE_string("negative_data_file", "./data/polarityData/tweets.neg", "Negative data source TRAINING SOURCE")

#Model params
tensorflow.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tensorflow.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tensorflow.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tensorflow.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep chance decimal (default: 0.5) rate = 1 - keep_prob")
tensorflow.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularisation lambda (default: 0.0)")

#Training params
tensorflow.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tensorflow.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tensorflow.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model and progress every x steps (default: 100)")
tensorflow.flags.DEFINE_integer("checkpoint_every", 100, "Save model in /ckeckpoints every x steps (default: 100)")
tensorflow.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints stored (default: 5)")

#Misc params
tensorflow.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tensorflow.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#assign tensorflow.flags.FLAGS to simply FLAGS
FLAGS = tensorflow.flags.FLAGS

#-=-=-=-=-=END OF DEFINITIONS=-=-=-=-=-#

def preprocess():
    """
    Data Pre-Processing,
    Loads all of the data into the model for training & evaluation,
    Returns XY vectors for train and dev,
    As well as the vocabulary processor,
    """
    #Load input Data from File
    print("Loading Data ...")
    xText, y = dataHelper.loadDataAndLabels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    #Build Vocab List
    maxDocLength = max([len(x.split(" ")) for x in xText])
    vocabProcessor = learn.preprocessing.VocabularyProcessor(maxDocLength)
    x = numpy.array(list(vocabProcessor.fit_transform(xText)))

    #Shuffle input data
    numpy.random.seed(10)
    indexShuffle = numpy.random.permutation(numpy.arange(len(y)))
    xShuffled = x[indexShuffle]
    yShuffled = y[indexShuffle]

    #Split input into train/test data
    devSample = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = xShuffled[:devSample], xShuffled[:devSample]
    y_train, y_dev = yShuffled[:devSample], yShuffled[:devSample]
    #delete unused input to save memory
    del x, y, xShuffled, yShuffled

    #Print vocab size and data splitm return data
    print("Vocab Size: {:d}".format(len(vocabProcessor.vocabulary_)))
    print("Train/Dev Split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocabProcessor, x_dev, y_dev

def train(x_train, y_train, vocabProcessor, x_dev, y_dev):
    """
    Procedure to Train Model,
    Writes to console and file accuracy and loss scores at each step,
    This can be read by Tensorboard,
    """

    with tensorflow.Graph().as_default():
        #initalise tensorflow session
        sessionConf = tensorflow.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )
        session = tensorflow.Session(config=sessionConf)
        with session.as_default():
            #initialise neural network
            neuralNet = TextNeuralNet(
                sequenceLength = x_train.shape[1],
                numClasses = 2,
                vocabSize = len(vocabProcessor.vocabulary_),
                embedSize = FLAGS.embedding_dim,
                filterSizes = list(map(int, FLAGS.filter_sizes.split(","))),
                numFilters = FLAGS.num_filters
            )

            #training procedure, defines steps and Adam Optimiser
            globalStep = tensorflow.Variable(0, name = "globalStep", trainable = False)
            optimiser = tensorflow.train.AdamOptimizer(1e-4)
            gradsVars = optimiser.compute_gradients(neuralNet.loss)
            trainOp = optimiser.apply_gradients(gradsVars, global_step = globalStep)

            #Setup output directory
            timestamp = str(int(time.time()))
            outputDir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(outputDir))

            #Loss & Accuracy Summaries
            lossSummary = tensorflow.summary.scalar("loss", neuralNet.loss)
            accSummary = tensorflow.summary.scalar("accuracy", neuralNet.accuracy)

            #Training Summaries
            trainSummaryOp = tensorflow.summary.merge([lossSummary, accSummary])
            trainSummaryDir = os.path.join(outputDir, "summaries", "train")
            trainSummaryWriter = tensorflow.summary.FileWriter(trainSummaryDir, session.graph)

            #Dev/Test Summaries
            devSummaryOp = tensorflow.summary.merge([lossSummary, accSummary])
            devSummaryDir = os.path.join(outputDir, "summaries", "dev")
            devSummaryWriter = tensorflow.summary.FileWriter(devSummaryDir, session.graph)

            #Setup Checkpoint directory, stores last checkpoints
            checkpointDir = os.path.abspath(os.path.join(outputDir, "checkpoint"))
            checkpointPrefix = os.path.join(checkpointDir, "model")
            if not os.path.exists(checkpointDir):
                os.makedirs(checkpointDir)
            saver = tensorflow.train.Saver()

            #Write vocab to file
            vocabProcessor.save(os.path.join(outputDir, "vocab"))

            #run session and initialise variables
            session.run(tensorflow.global_variables_initializer())

            def trainStep(xBatch, yBatch):
                """
                Individual Training Step
                Prints out timestamp, step number, and loss & accuracy values
                """
                #feed inputs into neural network
                feedDict = {
                    neuralNet.inputX: xBatch,
                    neuralNet.inputY: yBatch,
                    neuralNet.rate: 1 - FLAGS.keep_prob
                }

                #build step summary
                _, step, summaries, loss, accuracy = session.run(
                    [trainOp, globalStep, trainSummaryOp, neuralNet.loss, neuralNet.accuracy],
                    feedDict)

                #print timecode of step, step number, and loss/accuracy scores
                timeStr = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(timeStr, step, loss, accuracy))
                trainSummaryWriter.add_summary(summaries, step)

            def devStep(xBatch, yBatch, writer = None):
                """
                Individual Testing Step
                Prints out timestamp, step number, and loss & accuracy values
                """
                #feed inputs into neural network
                feedDict = {
                    neuralNet.inputX: xBatch,
                    neuralNet.inputY: yBatch,
                    neuralNet.rate: 0.0
                }

                #build step summary
                step, summaries, loss, accuracy = session.run(
                    [globalStep, devSummaryOp, neuralNet.loss, neuralNet.accuracy],
                    feedDict)

                #print timecode of step, step number, and loss/accuracy scores
                timeStr = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(timeStr, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            #Generate data batches
            batches = dataHelper.batchIterate(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            #iterate through data
            for batch in batches:
                xBatch, yBatch = zip(*batch)
                trainStep(xBatch, yBatch)
                currentStep = tensorflow.train.global_step(session, globalStep)
                #if on evaluation step...
                if currentStep % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    devStep(x_dev, y_dev, writer = devSummaryWriter)
                    print("")
                #if at a checkpoint...
                if currentStep % FLAGS.checkpoint_every == 0:
                    path = saver.save(session, checkpointPrefix, global_step = currentStep)
                    print("Saved Model Checkpoint to {}".format(path))

#run code
def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tensorflow.app.run()
#! /E:/Anaconda3 python
#This script was built so I could quickly process the sentiment 140 data into something that could be read by my own script
#It uses only 2 of the columns from the sentiment 140 data (tweet, sentiment classification) and puts them into CSV
import csv

with open("./data/trainingandtestdata/testdata.manual.2009.06.14.csv", "rt") as file:
    reader = csv.reader(file)
    posOutput = open("./data/trainingandtestdata/tweets.pos", "w+")
    negOutput = open("./data/trainingandtestdata/tweets.neg", "w+")

    for row in reader:
        if row[0] == "0":
            negOutput.write("{}\n".format(row[5]))
        if row[0] == "4":
            posOutput.write("{}\n".format(row[5]))
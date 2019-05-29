# ReadMe
# Twitter_SentimentAnalysis
Python Script to perform Sentiment Analysis of tweets
Twitter Sentiment Analysis using One-Dimensional Neural Networks
This zip folder contains:
 - Python Script for performiong Sentiment Analysis
 - Training & Evaluation Data used during the project
 - Training & Evaluation Run (TS: 1556277843)
   - Training Summaries readable via TensorBoard
   - Predictions.csv, a list of all of the predictions for the evaluation stage

If running via command line, make sure line 1 is pointing to the right position of the python env
Same goes for parameters pointing to training and eval files, make sure these are all correct before running
Running script can be done in both an IDE and via command line

Run Script on training.py first and wait for it to finish (or cancel early if too many epochs)
Then run evaluator.py to test

Dependencies can be found in requirements.txt
If using Anaconda:
```sh
conda install --file requirements.txt
```

# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math

def genProb(wordBag, smoothing_parameter):
    probList = {}
    totalWords = 0
    for word in wordBag.keys():     # Note that len(wordBag.keys()) = total number of word types
        totalWords += wordBag[word] # Total number of words in entire word bag.

    unk = math.log10(smoothing_parameter / (totalWords + smoothing_parameter * (len(wordBag.keys()) + 1))) # a / (n + a (V+ 1))
    
    for word in wordBag.keys():
        probList[word] = math.log10((wordBag[word] + smoothing_parameter) / (totalWords + smoothing_parameter * (len(wordBag.keys()) + 1)))

    return probList, unk

def isWordTooCommon(word):
    if len(word) <= 2 and word != 'no':
        return True
    elif word.lower() == 'the' or word.lower() == 'and' or word.lower() == 'this' or word.lower() == 'she':
        return True
    elif word.lower() == 'his' or word.lower() == 'her' or word.lower() == 'they' or word.lower() == 'them':
        return True
    elif word.lower() == 'that' or word.lower() == 'you' or word.lower() == 'are' or word.lower() == 'your':
        return True
    elif word.lower() == 'have' or word.lower() == 'has' or word.lower() == 'what' or word.lower() == 'too':
        return True
    elif word.lower() == 'also' or word.lower() == 'was':
        return True
    # if word.lower() == 'i' or word.lower() == 'the' or word.lower() == 'and' or word.lower() == 'this':
    #     return True
    # elif word.lower() == 'he' or word.lower() == 'she' or word.lower() == 'his' or word.lower() == 'her':
    #     return True
    # elif word.lower() == 'they' or word.lower() == 'them' or word.lower() == 'that' or word.lower() == 'a':
    #     return True
    # elif word.lower() == 'is' or word.lower() == 'an' or word.lower() == 'you' or word.lower() == 'it':
    #     return True
    # elif word.lower() == 'are' or word.lower() == 'your' or word.lower() == 'we' or word.lower() == 'by':
    #     return True
    # elif word.lower() == 'at' or word.lower() == 'be' or word.lower() == 'have' or word.lower() == 'has' or word.lower() == 'what':
    #     return True
    # elif word.lower() == 'in' or word.lower() == 'to' or word.lower() == 'of' or word.lower() == 'as' or word.lower() == 'or':
    #     return True
    # elif word.lower() == 'on' or word.lower() == 'also' or word.lower() == 'if' or word.lower() == 'me' or word.lower() == 'was':
    #     return True
    return False

def removeSingles(wordBag):
    newBag = {}
    for word in wordBag.keys():
        if wordBag[word] == 1:
            continue
        newBag[word] = wordBag[word]
    return newBag
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # Bag of words model
    smoothing_parameter = .007                      # override smoothing parameter
    posWordBag = {}
    negWordBag = {}
    for reviewIndex, review in enumerate(train_set):
        for word in review:
            if isWordTooCommon(word) == True:   # If word is too common, skip the word
                continue
            else:
                if train_labels[reviewIndex] == 1: # Positive
                    if word not in posWordBag.keys():
                        posWordBag[word] = 1
                    else:
                        posWordBag[word] += 1
                elif train_labels[reviewIndex] == 0: # Negative
                    if word not in negWordBag.keys():
                        negWordBag[word] = 1
                    else:
                        negWordBag[word] += 1
    
    posProbList, posUNK = genProb(posWordBag, smoothing_parameter)
    negProbList, negUNK = genProb(negWordBag, smoothing_parameter)

    # Done with training. Now development with MAP
    dev_labels = []
    for devRev in dev_set:
        reviewIsPos = math.log10(pos_prior)
        reviewIsNeg = math.log10(1 - pos_prior)
        for word in devRev:
            if isWordTooCommon(word) == True:   # If word is too common, skip the word
               continue
            else:
                if word in posProbList.keys():
                    reviewIsPos += posProbList[word]
                else:
                    reviewIsPos += posUNK
                if word in negProbList.keys():
                    reviewIsNeg += negProbList[word]
                else:
                    reviewIsNeg += negUNK
        if reviewIsPos < reviewIsNeg:
            dev_labels.append(0)
        else:
            dev_labels.append(1)

    # return predicted labels of development set
    return dev_labels

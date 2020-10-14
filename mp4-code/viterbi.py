"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

# Function made using the following: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
def getBestTag(dictionary):
    return max(dictionary, key=dictionary.get)

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wordTags = {}
    predicts = []

    for sentence in train:
        for index, wordTag in enumerate(sentence):
            if index == 0:          # The word is (START, START) so we skip it
                continue
            else:
                if wordTag[0] not in wordTags.keys():       # Word = (WORD, TAG) and not in dictionary
                    wordTags[wordTag[0]] = {}               # An empty dictionary
                    wordTags[wordTag[0]][wordTag[1]] = 1    # We say that the word has a tag with count 1
                else:
                    if wordTag[1] not in wordTags[wordTag[0]].keys():
                        wordTags[wordTag[0]][wordTag[1]] = 1
                    else:
                        wordTags[wordTag[0]][wordTag[1]] += 1

    for sentence in test:
        predictedSentence = []
        for index, word in enumerate(sentence):
            if index == 0:
                predictedSentence.append((word, 'START'))
            else:
                if word in wordTags.keys():
                    tag = getBestTag(wordTags[word])
                    predictedSentence.append((word, tag))
                else:
                    predictedSentence.append((word,'NOUN'))
        predicts.append(predictedSentence)
    return predicts


def getInitialProbs(tagDict, size, smoothing_parameter, tagList):
    probList = {}
    for tag in tagDict.keys():
        probList[tag] = math.log10((tagDict[tag] + smoothing_parameter) / (size + smoothing_parameter * (len(tagDict.keys()) + 1)))
    for tag in tagList:
        if tag not in probList.keys():
            probList[tag] = math.log10(smoothing_parameter / (size + smoothing_parameter * (len(tagDict.keys()) + 1)))
    return probList

def getTransitionProbs(twoTags, tags, smoothing_parameter, tagList):
    probList = {}

    for pair in twoTags:
        probList[pair] = math.log10((twoTags[pair] + smoothing_parameter) / (tags[pair[0]] + smoothing_parameter * (len(twoTags.keys()) + 1)))
        #probList[pair] = math.log10(twoTags[pair] / tags[pair[0]])
    for tagA in tagList:
        for tagB in tagList:
            if (tagA, tagB) not in probList.keys():
                probList[(tagA, tagB)] = math.log10(smoothing_parameter / (tags[tagA] + smoothing_parameter * (len(twoTags.keys()) + 1)))
    return probList

def getEmissionProbs(wordTags, tags, smoothing_parameter, tagList):
    probList = {}

    for word in wordTags.keys():
        probList[word] = {}
        for tag in wordTags[word].keys():
            probList[word][tag] = math.log10((wordTags[word][tag] + smoothing_parameter) / (tags[tag] + smoothing_parameter * (len(tags.keys()) + 1)))
            #probList[word][tag] = math.log10(wordTags[word][tag] / tags[tag])
    for word in wordTags.keys():
        for tag in tagList:
            if tag not in wordTags[word].keys():
                probList[word][tag] = math.log10((smoothing_parameter) / (tags[tag] + smoothing_parameter * (len(tags.keys()) + 1)))
    return probList

def getHapWords(wordTags, words):
    hapWords = []
    for word in words.keys():
        if words[word] == 1:
            tag = list(wordTags[word].keys())[0]
            hapWords.append((word, tag))
    return hapWords

def getHapProbs(hapWords, tagList, smoothing_parameter):
    tagOccurences = {}                         # Get the number of times a tag occurs in hapax word set.

    for wordTag in hapWords:
        if wordTag[1] not in tagOccurences.keys():
            tagOccurences[wordTag[1]] = 1
        else:
            tagOccurences[wordTag[1]] += 1
    # Total tags = length of hapwords
    probList = {}                # The probability list. 
    for tag in tagList:
        if tag in tagOccurences.keys():
            probList[tag] = (tagOccurences[tag] + smoothing_parameter) / (len(hapWords) + smoothing_parameter * (len(tagOccurences.keys()) + 1))
        else:
            probList[tag] = smoothing_parameter / (len(hapWords) + smoothing_parameter * (len(tagOccurences.keys()) + 1))
    return probList

def viterbi(train, test):
    '''
    TODO: implement the Viterbi algorithm.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    predicts = []
    initialTags = {}            # Counts the number of times a tag is the first word in a sentence
    tagOccurences = {}
    twoTagOccurences = {}
    wordTagOccurences = {}
    wordOccurences = {}

    initial_smoothing_parameter = 1         # Doesn't really change anything
    transition_smoothing_parameter = 1      # Might need tweaking
    emission_smoothing_parameter = .0001    # Smaller = better

    hap_smoothing_parameter = 1

    for sentence in train:
        for index, wordTag in enumerate(sentence):

            if wordTag[0] not in wordOccurences.keys():
                wordOccurences[wordTag[0]] = 1
            else:
                wordOccurences[wordTag[0]] += 1

            if index == 0:  # at index 0, wordTag = (START, START)
                if wordTag[1] not in initialTags.keys():
                    initialTags[wordTag[1]] = 1
                else:
                    initialTags[wordTag[1]] += 1
            #####
            if index < (len(sentence)-1):                     # Tag pairs. Condition ensures we don't work on the last word in sentence.
                tagA = wordTag[1]
                tagB = sentence[index+1][1]
                twoTags = (tagA, tagB)
                if twoTags not in twoTagOccurences.keys():
                    twoTagOccurences[twoTags] = 1
                else:
                    twoTagOccurences[twoTags] += 1
            else:                                           # Last word in sentence. Do the two-tag backwards.
                tagA = sentence[index-1][1]
                tagB = wordTag[1]
                twoTags = (tagA, tagB)
                if twoTags not in twoTagOccurences.keys():
                    twoTagOccurences[twoTags] = 1
                else:
                    twoTagOccurences[twoTags] += 1

            if wordTag[1] not in tagOccurences.keys():          # Number of times a tag occurs
                tagOccurences[wordTag[1]] = 1
            else:
                tagOccurences[wordTag[1]] += 1

            ######
            if wordTag[0] not in wordTagOccurences.keys():      # Number of times a (word, tag) pair occurs
                wordTagOccurences[wordTag[0]] = {}
                wordTagOccurences[wordTag[0]][wordTag[1]] = 1
            else:
                if wordTag[1] not in wordTagOccurences[wordTag[0]].keys():
                    wordTagOccurences[wordTag[0]][wordTag[1]] = 1
                else:
                    wordTagOccurences[wordTag[0]][wordTag[1]] += 1

    tagList = list(tagOccurences.keys())
    hapWords = getHapWords(wordTagOccurences, wordOccurences)
    hapProbs = getHapProbs(hapWords, tagList, hap_smoothing_parameter)

    initialProbs = getInitialProbs(initialTags, len(train), initial_smoothing_parameter, tagList)
    transitionProbs = getTransitionProbs(twoTagOccurences, tagOccurences, transition_smoothing_parameter, tagList)
    emissionProbs = getEmissionProbs(wordTagOccurences, tagOccurences, emission_smoothing_parameter, tagList)
    
    unkWordEmission = {}
    for tag in tagOccurences.keys():
        emission_smoothing_parameter = .0001 * hapProbs[tag]
        unkWordEmission[tag] = math.log10(emission_smoothing_parameter / (tagOccurences[tag] + emission_smoothing_parameter * (len(tagOccurences.keys()) + 1)))
    
    for sentence in test:
        # Trellis cells will be a tuple: (probability, prevTag)
        trellis = [[0 for x in range(len(tagList))] for y in range(len(sentence))]  # Creates 2D array. n x m. n = number of words in sentence. m = number of tags
        for wordIndex, word in enumerate(sentence):
            if wordIndex == 0:                              # First word in sentence.
                for tagIndex, tag in enumerate(tagList):
                    if word in wordTagOccurences.keys():
                        trellis[wordIndex][tagIndex] = (initialProbs[tag] + emissionProbs[word][tag], None)
                    else:
                        trellis[wordIndex][tagIndex] = (initialProbs[tag] + unkWordEmission[tag], None)
            
            else:                                           # Not the first word in the sentence.
                for tagBIndex, tagB in enumerate(tagList):          # Our previous tag
                    for tagAIndex, tagA in enumerate(tagList):      # Our current tag
                        if word in wordTagOccurences.keys():        # If we have seen the word before
                            probability = transitionProbs[(tagA, tagB)] + emissionProbs[word][tagB] + trellis[wordIndex-1][tagAIndex][0]
                        else:                                       # If we haven't seen the word before
                            probability = transitionProbs[(tagA, tagB)] + unkWordEmission[tagB] + trellis[wordIndex-1][tagAIndex][0]
                        
                        if tagAIndex == 0:                                              # If the previous tag is start
                            trellis[wordIndex][tagBIndex] = (probability, tagA)
                        
                        if trellis[wordIndex][tagBIndex][0] < probability:            # If our current probability is larger than what we already have
                            trellis[wordIndex][tagBIndex] = (probability, tagA)         # Update it.

        if len(trellis) == 0:
            predicts.append([])
            continue

        predSentence = [0 for x in range(len(sentence))]        # Holds tags for each word
        finalVals = trellis[len(trellis)-1]                     # The last column in the trellis. The last word with probabilities of each tag.
        maxLastVal = max(finalVals, key=lambda item: item[0])                            # The tuple with the largest probability
        bestTagIndex = finalVals.index(maxLastVal)                                       # The index of the largest probability is the best tag for last word.
        bestTag = tagList[bestTagIndex]                                                  # Best tag for last word
        predSentence[len(predSentence)-1] = bestTag                                     # Set tag for last word.
        for i in reversed(range(0, len(trellis)-1)):                                    # Start from second to last word to the second word.
            prevTagIdx = tagList.index(maxLastVal[1])
            maxLastVal = trellis[i][prevTagIdx]
            predSentence[i] = tagList[prevTagIdx]
        predicts.append(list(zip(sentence, predSentence)))

    return predicts
import re

def preprocessLogs(inputFileName, outputFileName):
	inputLogFile = open(inputFileName)
	outputFile = open(outputFileName, 'w')
	log = []
	flag = False
	docCount = 1
	for line in iter(inputLogFile):
		if line.strip().startswith('\"'):
			flag = True
		
		if flag is True:
			log.append(line.strip())
		
		if line.strip().endswith('\"'):
			logText = ' '.join(log)
			outputFile.write(str(docCount) + "\t" + logText + '\n')
			log = []
			flag = False
			docCount += 1
	inputLogFile.close()
	outputFile.close()

# Returns the list of stopwords
# TODO: Experiment with this as well. Stopwords in logs would be different as compared to normal documents
def getStopWords():
	stopwords = set()
	f = open("stopword.list")
	for line in iter(f):
		stopwords.add(line.strip())
	f.close()
	return stopwords
	
def isValidToken(token, stopWords):
	# TODO: Experiment with this
	return not (len(token) < 2 or (token in stopWords) or token.isdigit())

def processDocument(doc):
	# Tokenize the text.
	# TODO: Find various ways to tokenize text
	return re.sub('[^a-zA-Z]+', ' ', doc)

def generateGlobalWordFrequency(inputFileName):
	termDict = {}
	stopWords = getStopWords()
	inputFile = open(inputFileName)
	for line in iter(inputFile):
		processedLine = processDocument(line)
		tokens = processedLine.strip().split()
		documentVector = {}
		for token in tokens:
			if isValidToken(token, stopWords):
				# Global Word Frequency
				lowercasetoken = token.lower()
				if lowercasetoken in termDict:
					termDict[lowercasetoken] = termDict[lowercasetoken] + 1
				else:
					termDict[lowercasetoken] = 1
	inputFile.close()
	
	wordDfFile = open("processedData/word_df.txt", "w")
	wordDictFile = open("processedData/word_dict.txt", "w")
	wordId = 0
	termIDs = {}
	for word in termDict:
		wordDfFile.write(str(wordId) + ":" + str(termDict[word]) + "\n")
		wordDictFile.write(str(wordId) + ":" + word + "\n")
		termIDs[word] = wordId
		wordId += 1
	wordDfFile.close()
	wordDictFile.close()
	return termIDs

def generateDocumentVectors(inputFileName, termIDs):
	inputFile = open(inputFileName)
	outputFile = open("processedData/documentVectors.txt", "w")
	stopWords = getStopWords()
	for line in iter(inputFile):
		processedLine = processDocument(line)
		tokens = processedLine.strip().split()
		documentVector = {}
		for token in tokens:
			if isValidToken(token, stopWords):
				lowercasetoken = token.lower()
				if lowercasetoken in documentVector:
					documentVector[lowercasetoken] = documentVector[lowercasetoken] + 1
				else:
					documentVector[lowercasetoken] = 1
					
		tokenCount = []
		for token in documentVector:
			tokenCount.append(str(termIDs[token]) + ":" + str(documentVector[token]))
		outputFile.write(" ".join(tokenCount) + "\n")
	inputFile.close()
	outputFile.close()

preprocessLogs("data/log.csv", "data/documents.txt")
termIDs = generateGlobalWordFrequency("data/documents.txt")
generateDocumentVectors("data/documents.txt", termIDs)
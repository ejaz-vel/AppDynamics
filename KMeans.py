from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np
import math

dfFile = 'processedData/word_df.txt'
docFile = 'processedData/documentVectors_test.txt'
#docFile = 'processedData/documentVectors.txt'

# Count the number of lines in the DocumentFrequency(df) file
def getNumUniqueWords():
	f = open(dfFile)
	count = 0
	for line in iter(f):
		count += 1
	f.close()
	return count

# Count the number of lines in the DocumentVector(docVectors) file
def getNumDocuments():
	f = open(docFile)
	count = 0
	for line in iter(f):
		count += 1
	f.close()
        print "count ", count
	return count

def updateProbDisributionUsingSimilarity(documents, centroids):
	probDistribution = []
	runningSum = 0.0

	for n in range(len(documents)):
		maxSim = 0.0000001
		# Find the cluster with whom the documents n has maximum similarity
		for k in range(len(centroids)):
			sim = computeSimilarity(documents[n], centroids[k])
			if (sim.nnz == 0):
				sim = 0.0000001
			else:
				sim = sim.data[0]

			if maxSim < sim:
				maxSim = sim
		# Take invserse of the maximum Similarity and update the probability distribution
		# This makes sure documents with lower similarity have higher probability of being sampled
		probDistribution.append(1.0/maxSim)
		runningSum += 1.0/maxSim

	probDistribution[:] = [x / runningSum for x in probDistribution]
	return probDistribution

# documents: The list of document vectors in the corpus
# centroids: The list of centroids
def updateProbDisribution(documents, centroids):
	probDistribution = []
	runningSum = 0.0
	for n in range(len(documents)):
		minDist = 9999999.0
		#Find the cluster at the closest distance from the document n
		for k in range(len(centroids)):
			diff = documents[n]-centroids[k]
			dist = diff.multiply(diff).sum()
			if dist < minDist:
				minDist = dist
		# Update the probability distribution so that documents with higher distance
		# have higher probability of being sampled.
		probDistribution.append(minDist)
		runningSum += minDist
	probDistribution[:] = [x / runningSum for x in probDistribution]
	return probDistribution

# This is the entry point for initializaing the centroids for KMeans++
# documents: The list of document vectors in the corpus
# numCentroid: Number of centroids that need to be initialized
# useSimMetric: decides whether to use the distance metric or similarity metric
def initCentroidKMeanspp(documents, numCentroid, useSimMetric):
	centroids = []
	# Initially define the probability distribution giving equal probability of being sampled to each document.
	probDistribution = [1.0/len(documents)] * len(documents)
	values = range(len(documents))
	for k in range(numCentroid):
		# Sample a document from the distribution
		docID = np.random.choice(values, 1, p=probDistribution)[0]
		# Add this document vector to the centroids list. In this case the centroids will be non-sparse.
		centroids.append(documents[docID])
		print("Initialized centroid: " + str(k+1))

		if useSimMetric is True:
			probDistribution = updateProbDisributionUsingSimilarity(documents, centroids)
		else:
			probDistribution = updateProbDisribution(documents, centroids)
	return centroids

# This is the entry point for randomly initializaing the centroids for KMeans
def randomInitCentroid(numCentroid, vectorSize):
	centroids = []
	for i in range(numCentroid):
		# Randomly Initialize a centroid. This will be a non-sparse vector
		centroidVector = np.random.random(vectorSize)
		# Normalize the centroid Vector and add it to the list of centroids
		centroids.append(centroidVector * 1/np.linalg.norm(centroidVector))
	return centroids

# Calculate the inverse Document Frequence of each term in the corpus.
def getInverseDocumentFrequencyMap(numDocuments):
	# A map to store the IDF values for each term in the corpus
	dfMap = {}
	f = open(dfFile)
	for line in iter(f):
		df = line.split(":")
                if(df[0]=='1873'):
                    print "found ", df[0],"count",int(df[1]), numDocuments
		dfMap[df[0]] = numDocuments / int(df[1])
	return dfMap

# Initializing the document vectors by reading the data from the input document vector file
def getDocumentVectors(vectorSize):
	f = open(docFile)
	documents = []
	for line in iter(f):
		row = []
		col = []
		data = []
		wordFrequency = line.split()
		for word in wordFrequency:
			row.append(0)
			col.append(int(word.split(':')[0]))
			# Use only the term frequency to weigh each term in the document
			data.append(int(word.split(':')[1]))
		# Create the sparse document vector using the CSR_matrix data structure
		spVector = csr_matrix((data, (row, col)), shape=(1, vectorSize))
		# Normalize the document vector and add the normalized vector to the list of document vectors
		documents.append(spVector.multiply(1/np.sqrt(spVector.multiply(spVector).sum())))
	f.close()
	return documents

# Use the TF-IDF weights to initialize the document vectors
def getDocumentVectorsWithTFIDF(vectorSize):
	numDocuments = getNumDocuments()
	idf = getInverseDocumentFrequencyMap(numDocuments)

        #print "idf", idf['1873']
	f = open(docFile)
	documents = []
	for line in iter(f):
		row = []
		col = []
		data = []
		wordFrequency = line.split()
		for word in wordFrequency:
			row.append(0)
			col.append(int(word.split(':')[0]))
			# Use only the TF-IDF value to weigh each term in the document
                        if( idf[word.split(':')[0]] <=0):
                            print "Negative or zero", word, word.split(':'), idf[word.split(':')[0]]
			data.append(int(word.split(':')[1]) * math.log(idf[word.split(':')[0]]))
		# Create the sparse document vector using the CSR_matrix data structure
		spVector = csr_matrix((data, (row, col)), shape=(1, vectorSize))
		# Normalize the document vector and add the normalized vector to the list of document vectors
		documents.append(spVector.multiply(1/np.sqrt(spVector.multiply(spVector).sum())))
	f.close()
	return documents

# Compute Similarity between the document and the given centroid
def computeSimilarity(document, centroid):
	# Use the dot product as the similarity metric.
	# This is same as the cosine similarity because the document and centroid vectors have already been normalized before.
	sim = document.dot(centroid.T)[0]
	return sim

# Output the document-cluster assignment into the output file
def printDocumentClusters(clusterAssignment, outputFileName):
	docID = 1
	output = open(outputFileName, "w")
	for cluster in clusterAssignment:
		line = str(docID) + " " + str(cluster)
		output.write(line)
		output.write("\n")
		docID += 1
	output.close()

# Update the centroid by calculating the arithmetic mean of all the documents assigned to the cluster
def updateCentroid(documents, clusterAssignment, numClusters, vectorSize):
	centroidSum = np.zeros(shape=(numClusters,vectorSize))
	centroidCount = [0] * numClusters

	for n in range(len(clusterAssignment)):
		cluster = clusterAssignment[n]
		centroidSum[cluster] += documents[n]
		centroidCount[cluster] += 1

	centroids = []
	for k in range(numClusters):
		if centroidCount[k] > 0:
			# Find the arithmetic mean
			centroidVector = centroidSum[k] / centroidCount[k]
		else:
			# Empty Cluster! Randomly initialize the centroid in this case
			centroidVector = np.random.random((1, vectorSize))
		# Normalize the centorid before updating the centroids list
		centroids.append(centroidVector * 1/np.linalg.norm(centroidVector))

	return centroids

# Find the similarity of the documents with the cluster it is assigned to
def computeClusteringScore(clusterAssignment, centroids, documents):
	#Calculate Average Similarity of each Cluster
	centroidCount = [0] * len(centroids)
	centroidSimilarityCount = [0] * len(centroids)

	for n in range(len(clusterAssignment)):
		cluster = clusterAssignment[n]
		centroidCount[cluster] += 1
		centroidSimilarityCount[cluster] += computeSimilarity(documents[n], centroids[cluster])

	total = 0.0
	for k in range(len(centroids)):
		if centroidCount[k] == 0:
			avgSimilarity = 0
		else:
			avgSimilarity = centroidSimilarityCount[k] / centroidCount[k]
		total += avgSimilarity
	total = total / len(centroids)
	print "Average Similarity of the clusters: " + str(total)
	return total

# This is the main method that serves as the entry point to all the functionality
# numClusters: The number of clusters to be learnt
# useTFIDFWeight: This decides whether we need to use the TF-IDF weight or not while creating the document vector
# useKMeanspp: This decides whether we need to use KMeans++ or not to initialize the clusters
# useSimMetric: If we are using KMeans++, this parameter decides whether to use distance or similarity metric

def findClusterAssignment(numClusters, useTFIDFWeight=False, useKMeanspp=False, useSimMetric=False):
	vectorSize = getNumUniqueWords()
	documents = []
	centroids = []

	if useTFIDFWeight is True:
		print("Using TF-IDF weights for Document Vectors")
		documents = getDocumentVectorsWithTFIDF(vectorSize)
	else:
		print("Using TF weights for Document Vectors")
		documents = getDocumentVectors(vectorSize)

	if useKMeanspp == True:
		print("Using KMeans++ to initialize centroids")
		centroids = initCentroidKMeanspp(documents, numClusters, useSimMetric)
	else:
		print("Randomly initializing centroids")
		centroids = randomInitCentroid(numClusters, vectorSize)

	numDocs = len(documents)
	previousClusterAssignment = [-1] * numDocs
	clusterAssignment = [0] * numDocs

	numIterations = -1
	print("Running Loyd's Algorithm for clustering")
	while ( np.array_equal(previousClusterAssignment,clusterAssignment) == False):
		print("Iteration: " + str(numIterations))
		previousClusterAssignment = clusterAssignment[:]
		for n in range(numDocs):
			maxSim = 0.0
			cluster = clusterAssignment[n]
			for k in range(numClusters):
				sim = computeSimilarity(documents[n], centroids[k])
				if sim > maxSim:
					maxSim = sim
					cluster = k
			clusterAssignment[n] = cluster
		numIterations += 1
		centroids = updateCentroid(documents, clusterAssignment, numClusters, vectorSize)
	similarity = computeClusteringScore(clusterAssignment, centroids, documents)
	return clusterAssignment, centroids, similarity


if __name__ == "__main__":
    ca,centroids,simi = findClusterAssignment(3,True,True, True)
    printDocumentClusters(ca, 'output/assign.txt')

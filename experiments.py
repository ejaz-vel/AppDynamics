import KMeans

# Runs experiments for KMeans
# You have the option of choosing to use TF-IDF scores or not. 
def runExperimentsForKMeans(numClusters, numExperiments, useTFIDF=True):
	maxSim = 0
	experiment = 0
	for i in range(numExperiments):
		print "\nExperiment " + str(i) + " for " + str(numClusters) + " clusters"
		[clusterAssignment, similarity] = KMeans.findClusterAssignment(numClusters, useTFIDF)
		outputFileName = "output/KMeans" + str(numClusters) + "-" + str(i)
		KMeans.printDocumentClusters(clusterAssignment, outputFileName)
		if maxSim < similarity:
			maxSim = similarity
			experiment = i
	print "Max Similarity over " + str(numExperiments) + " Experiments: " + str(maxSim)
	return maxSim, experiment

# Runs experiments for KMeans++
# You have the option of choosing to use TF-IDF scores or not. 
# Also whether to use similarity metric to initialize the clusters or not
def runExperimentsForKMeansPP(numClusters, numExperiments, useTFIDF=True, useSimMetric=True):
	maxSim = 0
	experiment = 0
	for i in range(numExperiments):
		print "\nExperiment " + str(i) + " for " + str(numClusters) + " clusters"
		[clusterAssignment, similarity] = KMeans.findClusterAssignment(numClusters, useTFIDF, True, useSimMetric)
		outputFileName = "output/KMeanspp" + str(numClusters) + "-" + str(i)
		KMeans.printDocumentClusters(clusterAssignment, outputFileName)
		if maxSim < similarity:
			maxSim = similarity
			experiment = i
	print "Max Similarity for " + str(numExperiments) + " Experiments: " + str(maxSim)
	return maxSim, i

# This routine helps to find the number of clusters
def findBestCluster(useKMeansPP = False, useTFIDF = True):
	clusterSimilarity = []
	minClusters = 2
	maxClusters = 10
	previousSim = 0.0
	currentSim = 0.0
	
	# Find the average similarity of all the clusters
	for clusterSize in range(minClusters, maxClusters, 1):
		previousSim = currentSim
		if useKMeansPP is True:
			currentSim = runExperimentsForKMeansPP(clusterSize, 5, useTFIDF)[0]
		else:
			currentSim = runExperimentsForKMeans(clusterSize, 5, useTFIDF)[0]
		
		clusterSimilarity.append(currentSim)
		
		# We have found the optimal number of clusters once, similarity of these clusters start increasing very slowly
		if currentSim - previousSim < 0.015:
			diff = currentSim - previousSim
			if diff > 0:
				print "Optimal Number of Clusters: " + str(clusterSize)
			else:
				print "Optimal Number of Clusters: " + str(clusterSize-1)
			break
	return clusterSimilarity
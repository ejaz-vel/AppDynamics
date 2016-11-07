import sys

clusterAssignment = {}
bestClusterFile = sys.argv[1]
inputFile = open(bestClusterFile)
for line in iter(inputFile):
	data = line.split()
	clusterAssignment[int(data[0])] = int(data[1])
inputFile.close()

numCluster = int(sys.argv[2])
fileMap = {}
for cluster in range(numCluster):
	fileMap[cluster] = open("cluster" + str(cluster), "w")

docFileName = "data/documents.txt"
docFile = open(docFileName)
for line in iter(docFile):
	data = line.split("\t")
	cluster = clusterAssignment[int(data[0])]
	fileMap[cluster].write(data[1].strip() + "\n")
docFile.close()

for cluster in range(numCluster):
	fileMap[cluster].close()
"""
Algorithm to cluster the streaming data points.

Problems/task to look at :
1) Initial cluster formation (Before we kick in the algorithm - Assumed already done
2)
3)

Concept/Algorithm :

"""

"""
This function will generate the initial data based on the number of cluster provided
"""
import numpy as np
import matplotlib.pyplot as plt

# Data points generated.

min_density = 0 # min number of points present in the smallest cluster
curr_density = np.zeros((1,1)) # num points in the middels histogram bin (ambigous points)

n_samples = 1000 # number of samples in each cluster of inital points
streaming_samples = 700 # number of streaming samples requested

# Initializing the size of the clusters
data = np.zeros((n_samples,2))
streaming_data = np.zeros((streaming_samples,2))

"""
Points with good GMM i.e 0.8 and above are considered as good or non ambigous points
Points with middle (0.3-0.7) distribution are ambigous points and will be put into
bad_point_idx bucket.
"""
good_points_idx = [] # These points will be defined to the existing cluster
bad_point_idx = [] # These points will be reclustered

"""--------Hyper parameters----------------------
Threshold: Values between 0-1 showing when to trigger the re-run algorithm.
% of min_density(number of points in the smallest cluster) is the triggering point
Low bin : bin index below this are good points(high confidence on some cluster)
high bin : Bin index above this are good points(high confidence on some cluster)
Other than that, the points are considered as ambiguous and will add to the count
of points which will helps in triggering the re-clustering algorithm.
"""
density_threshold = .3
low_bin = 0.2
high_bin = 0.8

bins = np.zeros((10,1))

"""
Generate the toy data which is forming two clusters i.e 2D Gaussian function
"""
def generate_toy_data():

    print("Generating toy dataset !")
    # Generate 2D Gaussian distribution
    cluster_1_data = np.random.multivariate_normal(np.array([2, 6]), np.array([[0.5, 0], [0, 0.5]]), n_samples)
    cluster_2_data = np.random.multivariate_normal(np.array([8, 1]), np.array([[0.5, 0], [0, 0.5]]), n_samples)
    data = np.concatenate((cluster_1_data, cluster_2_data), axis=0)
    print(data.shape)

    min_density = min(cluster_1_data.size,cluster_2_data.size)

    plt.plot(cluster_1_data[:, 0], cluster_1_data[:, 1], '*', color='r')
    plt.plot(cluster_2_data[:, 0], cluster_2_data[:, 1], '*', color='g')
    plt.title(" Initial Data (2D Gaussian) : 1000 points each cluster")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()

"""
Generate the streaming data i.e.
Case 1 : Where the data are overlapping with the initial cluster.
Case 2 : Data is forming the new cluster/outliers
"""
def generate_streaming_data(case_num):

    print("Generating streaming dataset for case# :",case_num)

    # First case in which the data points are forming a new clusters
    if case_num==1:
        streaming_data = np.random.multivariate_normal(np.array([8,6]), np.array([[0.5, 0], [0, 0.5]]), n_samples)

    # Second case : new points are lying over to the existing clusters
    if case_num==2:
        stream_1_data = np.random.multivariate_normal(np.array([2, 6]), np.array([[0.5, 0], [0, 0.5]]), 300)
        stream_2_data = np.random.multivariate_normal(np.array([8, 1]), np.array([[0.5, 0], [0, 0.5]]), 300)
        streaming_data = np.concatenate((stream_1_data, stream_2_data), axis=0)
        np.random.shuffle(streaming_data)

    print('Streaming_data : ',streaming_data.dtype,streaming_data.shape)

    """
    # Plot the streaming points
    plt.plot(streaming_data[:, 0], streaming_data[:, 1], '*', color='b')
    plt.title(" Streaming data points")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()
    """
    return streaming_data

def print_inital_clusters():
    print("Getting clusters centroid :")
    print("Cluster 1: 2,6")
    print("Cluster 2: 8,1")
    #kmeans = KMeans(init='k-means++', n_clusters=2, n_init=2)
    #kmeans.fit()

"""
Go through the GMM distribution of this points and add it to the bin density of histogram
0-0.1 -> bin_idx = 0 -> bins[0]++
"""
def add_to_bin(gmm_values):
    for i in range(len(gmm_values)):
        curr_prob = gmm_values[i]
        bin_idx = int(curr_prob*10)
        bins[bin_idx,1] = bins[bin_idx,1] + 1
        if (bin_idx > low_bin) & (bin_idx<high_bin):
            curr_density[0,0] = curr_density[0,0]+1

# Check if the density of ambiguous point is more than some %(threshold) of the minimum density
def threshold_check():
    if curr_density >= density_threshold * min_density:
        print("Ambiguous points density is more than threshold, Kick in re-clustering !!")
        plot_bin_histogram()
        return True
    else:
        return False

# Plot the histogram of current bin distribution
def plot_bin_histogram():
    plt.hist(bins, 10, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Bin index ')
    plt.ylabel('Points count')
    plt.title('Bins histogram')
    plt.show()

"""
Given the point of d dimension, return the numpy array stating the probability of
belonging to the particular cluster.
The output array has the dimension of [k,1], where k is the number of clusters
already present. The sum of the array returned should sum up to 1.
"""
def get_gmm_array(point):
    arr = np.zeros((2,1))
    arr[0,1] = 0.4
    arr[1,1] = 0.6
    # GMM will be called on the point array
    return 1

# Main function
def main():
    print("Re-clustering algorithm !")

    # Generate the toy data set with two clusters.
    generate_toy_data()

    # Print the information of initial clusters
    print_inital_clusters()

    # Generate streaming data (Case 1)
    generate_streaming_data(1)

    # Go through streaming data points one by one
    for i in range(len(streaming_data)):

        # Step 1 : Get the GMM array of this point
        gmm_array = get_gmm_array(streaming_data(i))

        # Step 2 : Add the corresponding probabilites to the bin
        add_to_bin(gmm_array)

        # Step 4 : Plot the histogram periodically to check the density
        if i%100==0:
            plot_bin_histogram()

        # Step 3 : Check if the Ambiguous points density is higher thn threshold
        if threshold_check():
            break

# Call the main function which will do all other jobs.
if __name__ == '__main__':
    main()
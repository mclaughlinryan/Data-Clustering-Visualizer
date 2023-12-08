# Data-Clustering-Visualizer

This is a graphical user interface application that allows the user to visualize the clustering of data through selection from several different clustering algorithms as well as side-by-side comparison of different data clusterings.

#### Data File Format

The file containing the data must have data points with the same set of features and the features must be comma separated. The data points could also have classifications assigned to them as the last value in the file line.

#### Application

<img width="600" alt="data clustering visualizer" src="https://github.com/mclaughlinryan/Data-Clustering-Visualizer/assets/150348966/ce8b8d95-c2b4-41bc-9249-5a3d769d93a7">

&nbsp;

Clustering algorithms that can be used:
- K-means
- Mean shift
- DBSCAN
- HDBSCAN
- Gaussian mixture models
- Agglomerative clustering
- Affinity propagation
- Spectral clustering
- BIRCH
- OPTICS

The data clustering is done on numeric values so if features contain non-numeric values there are a few options for how they are handled:
- Assigning all non-numeric values to 0
- Assigning a number to each different non-numeric value
- Excluding data points with non-numeric values
- Excluding features with non-numeric values

Different data clusterings can also be viewed side by side:

<img width="900" alt="data clustering visualizer 2" src="https://github.com/mclaughlinryan/Data-Clustering-Visualizer/assets/150348966/15f7a43b-394c-484b-ae1e-8678b314eaee">

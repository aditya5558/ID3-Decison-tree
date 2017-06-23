# ID3-Decision-Tree


ID3 is a simple decision learning algorithm developed ID3 constructs decision tree by employing a top-down, greedy search through the given sets of training data to test each attribute at every node. It uses statistical property call information gain to select which attribute to test at each node in the tree. Information gain measures how well a given attribute separates the training examples according to their target classification. Decision tree algorithms are a method for approximating discrete-valued target functions, in which the learned function is represented by a decision tree. These kinds of algorithms are famous in inductive learning and have been successfully applied to a broad range of tasks. 


## The Algorithm

The ID3 algorithm builds decision trees using a top-down, greedy approach. Briefly, the steps to the algorithm are:
 
1) Start with a training data set, which we will call S. It should have attributes and classifications.
 
2) Determine the best attribute in the data set S. How to pick the best attribute is explained next.
 
3) Split S into subsets that correspond to the possible values of the best attribute.
 
4) Make a decision tree node that contains the best attribute.
 
5) Recursively make new decision tree nodes with the subsets of data created in step 3. Attributes cannot be reused. If a subset of data agrees on the classification, choose that. If there are no more attributes to split on, choose the most popular classification.
 

The ID3 algorithm for building a decision tree has been implemented on the famous Titanic Dataset. Clustering of input can be done if the attributes take continuous values to improve the accuracy.


## Files and their functionality

| File | Description |
|------|--------------|
|data_preProcess.cpp | Gets the input in the csv file to the required format|
|id3_general.cpp | Implements the ID3 algorithm |
|Compare_accuracy.py | Compares the performance of our ID3 implmentation with other contemporary classifiers|
|cluster.py | Clusters the input |

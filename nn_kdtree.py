STUDENT_ID = 'a1821415'
DEGREE = 'UG'
# From the given training data, our goal is to learn a function that can predict the wine quality rating of a wine sample, based on the features. 
# In this assignment, the predictor function will be constructed as a k-d tree. 
# Since the features are continuously valued, you shall apply the k-d tree algorithm for continuous data, as outlined in Algorithm 1. 
# It is the same as taught in the lecture. Once the tree is constructed, you will search the tree to find the 1-nearest neighbour of a query point and label the query point. 
# Please refer to the search logic taught in the lecture to write your code for the 1NN search.


import sys
import math
import numpy as np

class Node:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

def BuildKdTree(P, D, start_dim):
# Require: A set of points P of M dimensions and current depth D.
    if len(P) == 0:
        return None
    elif len(P) == 1:
        # create new node node
        # node.d <- d
        # node.val <- d
        # node.point <- current point
        # return node
        return Node(P[0], (start_dim + D) % len(P[0])-1)
    else:
        #d <- D mod M
        # val <- Median value along dimension among points in P.
        # node.d <- d
        # node.val <- d
        # node.point <- point at the median along dimension d
        # node.left <- BuildKdTree(points in P for which value at dimension d is less than or equal to val, D+1)
        # node.right <- BuildKdTree(Points in P for which value at dimension d is greater than val, D+1)
        #return node
        
        M = len(P[0]) - 1 #exclude label
        axis = (start_dim + D) % M
        P.sort(key=lambda x: x[axis])
        median = len(P) // 2
        med_val = P[median][axis]
        
        lft = []
        rht = []
        for p in P:
            if p[axis] < med_val:
                lft.append(p)
            elif p[axis] > med_val:
                rht.append(p)
            else:  # equal
                lft.append(p)

        # Edge case protection: ensure progress
        if len(lft) == len(P) or len(rht) == len(P):
            # Use slicing as fallback split
            lft = P[:median]
            rht = P[median+1:]
        
        node = Node(
            point = P[median],
            axis = axis,
            left = BuildKdTree(lft, D+1, start_dim),
            right = BuildKdTree(rht, D+1, start_dim)
        )
        
        if D == 0:
            indent = "." * start_dim
            print(indent + "l" + str(len(lft)-1))
            print(indent + "r" + str(len(rht)))
        
        return node

def euclidian_d(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

def kd_tree_1nn_search(root, query, best=None, best_dist=float('inf')):
    if root is None:
        return best, best_dist

    point = root.point
    axis = root.axis
    dist = euclidian_d(query, point[:-1])

    if dist < best_dist:
        best = point
        best_dist = dist

    diff = query[axis] - point[axis]
    close, away = (root.left, root.right) if diff <= 0 else (root.right, root.left)

    best, best_dist = kd_tree_1nn_search(close, query, best, best_dist)

    if abs(diff) < best_dist:
        best, best_dist = kd_tree_1nn_search(away, query, best, best_dist)

    return best, best_dist

# Sorting (at each level) may not be necessary depending on your implementation. 
# Google/ChatGPT for "Quickselect" if you are interested in implementing a K-d tree without doing the sorting. 
# Full pre-sorting before building the k-d tree is another possible solution, but it is usually more complex.

#Deliverable

#Program must run as |$ python3 nn_kdtree.py [train] [test] [dimension]
# where
# [train] specifies the path to the file containing a set of training data
# [test] specifies the path to a file containing a set of testing data
# [dimension] is used to decide which dimension to start the algorithm (BuildKdTree algorithm)

# Given the inputs, your program must construct a k-d tree (follow the psuedocode) using the training data, then predict the quality rating of each of the wine samples in the testing data.
# Your program must print two sections of content to the standard output:
    # 1. The number of points in the left and right trees after the first split of the k-d tree
    # e.g. :
        #...l4877
        #r412
        # where the number of '.' to print is indicated by D in above algorithm l and r define which side and the number is the number of points in each subtree
        
    #2. TThe list of predicted wine quality ratings, vertically based on the order in which the testing cases appear in [test]
    
#your program must be able to terminate withing 600 seconds of the sample data being given

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
    return [list(map(float, line.strip().split())) for line in lines]

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 nn_kdtree.py [train] [test] [dimension]")
        return

    train_file, test_file, start_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    kd_tree = BuildKdTree(train_data, 0, start_dim)

    for test_point in test_data:
        nearest, _ = kd_tree_1nn_search(kd_tree, test_point)
        print(int(nearest[-1]))

if __name__ == "__main__":
    main()
# 🍷 Wine Quality Prediction using k-d Tree (1-Nearest Neighbour)

This project implements a **k-d tree** for continuous data to perform **1-Nearest Neighbour (1NN) classification** on a wine quality dataset.

The goal is to predict the quality rating of wine samples based on their feature values using an efficiently constructed k-d tree.

---

## Assignment Objective

From the given training dataset:

- Construct a **k-d tree** using continuous-valued features.
- Implement a **1-nearest neighbour search** algorithm.
- Predict the quality rating for each test sample.
- Follow the k-d tree algorithm as taught in lectures.
- Ensure the program runs within **600 seconds** on provided sample data.

---

## Algorithm Overview

### Building the k-d Tree

The tree is built recursively using:

- Axis selection:
  

axis = (start_dimension + depth) mod M


- Median split along the selected axis.
- Left subtree → values ≤ median
- Right subtree → values > median

Where:
- `M` = number of feature dimensions (excluding label)

Each node contains:
- The full data point (features + label)
- The splitting axis
- Left and right children

---

### 1-Nearest Neighbour Search

To classify a query point:

1. Traverse down the tree to the leaf node along the splitting axis.
2. Track the current best (closest) distance.
3. Backtrack and check opposite branch **only if necessary**:
 

|difference along split axis| < current_best_distance


Distance metric used:


Euclidean Distance


The predicted wine quality is the label of the nearest neighbour.

---

## File Structure


nn_kdtree.py # Main implementation (tree building + 1NN search)


---

## Usage


python3 nn_kdtree.py [train] [test] [dimension]


### Arguments

| Argument | Description |
|----------|------------|
| `[train]` | Path to training data file |
| `[test]` | Path to testing data file |
| `[dimension]` | Starting dimension index for first tree split |

---

## Input Format

Both training and test files:

- First line: Header (ignored)
- Remaining lines: Space-separated numerical values

Training data:

feature1 feature2 ... featureM label


Test data:

feature1 feature2 ... featureM


---

## Output Format

The program prints **two sections**:

---

### First Split Summary

After the initial split of the k-d tree, print:


...l<number_of_points_in_left_subtree>
...r<number_of_points_in_right_subtree>


- Number of `.` characters equals the starting dimension.
- `l` = left subtree
- `r` = right subtree

Example:


...l4877
r412


---

### Predicted Labels

Predicted wine quality values printed vertically in the same order as test data:


6
5
7
4
...


---

## Implementation Details

- Median split chosen via sorting along selected axis.
- Edge-case fallback implemented if all points fall to one side.
- Recursive tree construction.
- Recursive backtracking 1NN search.
- Euclidean distance metric.
- Label excluded from distance calculation.
- Designed to terminate within required time constraints.

---

## Complexity Overview

- Tree Construction:
  - O(N log N) average (due to recursive median splitting)
- 1NN Search:
  - O(log N) average case
  - O(N) worst case

---

## Summary

This project demonstrates:

- Practical implementation of k-d tree construction
- Recursive data structures
- Nearest neighbour search with pruning
- Efficient spatial partitioning for continuous data

---
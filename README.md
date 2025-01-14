# BinaryData_HighOrderInteractions_GreedyAlgo
This algorithm performs a greedy search to find the most relevant patterns of binary data. Uncovered patterns can be of any order.

## Requirements

The code uses the C++11 version of C++.

It also uses the two header-only C++ libraries: Eigen and LBFGS++ (L-BFGS solver).
Remember to specify the paths to the two libraries to your compiler (see option `-I` below).

On the machine the modified Greedy algorithm was made, eigen3 was only available by adding the eigen folder to the main directory. 
The eigen3.zip is included which needs to be unzipped and put in the same directory, or change the way Eigen is included.

**To compile:**  `g++ -std=c++11 -I ./eigen3/ -I ./LBFGSpp-master/include -O2 main.cpp ReadData_DataBasisTransform.cpp IndepModel.cpp Models.cpp ModelStatistics.cpp BoltzmannLearning.cpp HeuristicAlgo.cpp BestBasis.cpp`

**To execute:** `./a.out`

## Examples

All the functions that can be used from `int main()` are declared at the beginning of the `main.cpp` file and described below (to come).

For hands-on and simple tests of the program, check the examples in the `int main()` function of the `main.cpp` file.

## License
This is an open source project under the GNU GPLv3.



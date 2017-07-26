**Model**: XOR
**Input**: x = (bool, bool)
**Output**: y = bool
**Objective Function**: *f(x, y) = 1 iff (x, y) = (0, 1) or (1, 0) and f(x, y) = 0 otherwise*
**Training Data**: *((x, y), f(x, y)) = ((0, 1), 1), ((1, 0), 1), ((0, 0), 0), ((1, 1), 0)*
**Model**: The network has the following dimension
       *2 -> 8 -> 8 -> 8 -> 1 -> 1*
**Nodes**: 
1. *First* node is a linear transformation of the input with dimensions *2 -> 8*.
2. *Second* node adds an 8 dimensional bias vector to the resulting 8  dimensional vector, resulting in an 8 dimensional vector.
3. *Third* node applies tanh to the entries of the resulting 8 dimensional vector, resulting in an eight dimensional vector.
4. *Fourth* node is a linear transformation with dimensions *8 -> 1*.
5. *Fifth* node adds a constant to the resulting scalar, resulting in a scalar.
6. *Sixth*, there is an optional tanh transoformation at the end that can be enabled by setting a variable in the script as the two different structures might result in different performance.

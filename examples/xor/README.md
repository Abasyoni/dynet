Model: XOR
Input: bool x bool
Output: bool
Objective Function: f(x, y) = 1 iff (x, y) = (0, 1) or (1, 0) and 
                    f(x, y) = 0 otherwise

Training Data: ((x, y), f(x, y)) = ((0, 1), 1), ((1, 0), 1), ((0, 0), 0), ((1, 1), 0)

Model: The network has the following dimension
       2 -> 8 -> 8 -> 8 -> 1 -> 1

Nodes: First node is a linear transformation of the input with dimensions 2 -> 8
       Second node adds an 8 dimensional bias vector to the resulting 8 dimensional vector, resulting in an 8 dimensional vector
       Third node applies tanh to the entries of the resulting 8 dimensional vector, resulting in an eight dimensional vector.
       Fourth node is a linear transformation with dimensions 8 -> 1
       Fifth node adds a constant to the resulting scalar, resulting in a scalar.
       Sixth, there is an optional tanh transoformation at the end that can be enabled by setting a variable in the script as the two different structures might result in different performance. 


Training: To train, we feed the four data p1oints over and over and adjust the weights using the backpropogation algorithm, or by calling loss.backward(). It only took 15 iterations before converging to the optimal weights, reaching loss of 0.00. 



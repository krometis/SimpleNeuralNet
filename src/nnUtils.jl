#Some useful functions

#Define the sigmoid function & its derivative
sigmoid(x)  = 1.0 / (1.0 + exp(-x));
dsigmoid(x)  = (1.0 / (1.0 + exp(-x))^2)*exp(-x);

#Test start to finish

using SimpleNeuralNet
using LinearAlgebra #for norm()
using Printf

#Build the neural net
nNeurons = [4; 5; 2]; #number of neurons by layer
nn = neuralNet(nNeurons);

#Define some important functions
SimpleNeuralNet.nnAct(z) = sigmoid.(z); #activation function (assume the same everywhere)
SimpleNeuralNet.nnDAct(z) = dsigmoid.(z); #derivative of activation function (assume the same everywhere)
SimpleNeuralNet.nnCost(datay,y) = 0.5*norm(y-datay,2)^2;
SimpleNeuralNet.nnDCost(datay,y) = y - datay;

#Initialize the weights and biases to random values
nn.w = rand(length(nn.w));
nn.b = rand(length(nn.b));

#Define the training parameters
lr=1.0;      #learning rate
epochs=1000; #epochs

#Define the dataset
nData = 10;
datax = rand(nn.nNeurons[1],nData);   #draw input dataset at random
datay = rand(nn.nNeurons[end],nData); #draw output dataset at random

#Compute the initial costs
c0 = [ nnCost(datay[:,d],nnForward(nn,datax[:,d])) for d=1:size(datax,2) ];

#Train
nnTrain(nn,datax,datay,epochs,lr;verbose=true);

#Compute the final costs
cf = [ nnCost(datay[:,d],nnForward(nn,datax[:,d])) for d=1:size(datax,2) ];

#Print a comparison of initial/final costs
for d=1:length(cf)
    @printf("dataset %3d: initial cost=%10.8f final cost=%10.8f\n",d,c0[d],cf[d]);
end

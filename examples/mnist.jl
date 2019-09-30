#Test start to finish

using SimpleNeuralNet
using LinearAlgebra #for norm()
using Printf
using StatsBase
using Statistics

#Define the training parameters
lr=0.1;      #learning rate
epochs=10;   #epochs
bs=100;      #batch size

#Define some important functions
SimpleNeuralNet.nnAct(z) = sigmoid.(z); #activation function (assume the same everywhere)
SimpleNeuralNet.nnDAct(z) = dsigmoid.(z); #derivative of activation function (assume the same everywhere)
SimpleNeuralNet.nnCost(datay,y) = 0.5*norm(y-datay,2)^2;
SimpleNeuralNet.nnDCost(datay,y) = y - datay;

#Get the mnist dataset
using MLDatasets
train_x, train_y = MNIST.traindata();
datax = Float64.(MNIST.convert2features(train_x));
#datay = Float64.(train_y);
datay = zeros(10,length(train_y));
for i=0:9; datay[i+1,:] = (train_y .== i); end

#Build the neural net
nIn  = size(datax,1);
nOut = size(datay,1);
nNeurons = [nIn; 300; nOut]; #number of neurons by layer
nn = neuralNet(nNeurons);

#Initialize the weights and biases to random values
nn.w = rand(length(nn.w));
nn.b = rand(length(nn.b));
#Normalize
nn.w ./= sum(nn.w);

#Compute performance
outputs = nnComputeOutputs(nn,datax);
nnComputeCosts(outputs,datay;print=true);
bestGuess=zeros(Int64,size(datax,2)); for i=1:size(datax,2); bestGuess[i] = findmax(outputs[:,i])[2]-1; end
@printf("Error rate: %8.6f\n",mean( bestGuess .!= train_y ));

#Train
nnTrain(nn,datax,datay,epochs,lr;batchSize=bs, verbose=true);

#Compute performance
outputs = nnComputeOutputs(nn,datax);
nnComputeCosts(outputs,datay;print=true);
bestGuess=zeros(Int64,size(datax,2)); for i=1:size(datax,2); bestGuess[i] = findmax(outputs[:,i])[2]-1; end
@printf("Error rate: %8.6f\n",mean( bestGuess .!= train_y ));


module SimpleNeuralNet

using LinearAlgebra
using Printf
using HDF5
using StatsBase

export neuralNet;
export nnForward;
export nnBackPropagate;
export nnTrain;
export nnLoad;
export nnSave;

#useful functions
export sigmoid;
export dsigmoid;

#function stubs
export nnAct;
export nnDAct;
export nnCost;
export nnDCost;

include("neuralNet.jl");
include("nnForward.jl");
include("nnBackPropagate.jl");
include("nnTrain.jl");
include("nnLoad.jl");
include("nnSave.jl");
include("nnStubs.jl");
include("nnUtils.jl");

end #end module

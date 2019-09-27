#neuralNet type: Defines the structure of the neural net
#and allocates arrays that will be used over and over again
#during training
mutable struct neuralNet
    nNeurons::Array{Int64} #number of neurons by level (including input and output levels)
    nLevels::Int64         #number of levels
    
    x::Array{Float64} #activations (one per neuron, including inputs and outputs)
    w::Array{Float64} #weights
    b::Array{Float64} #biases (one per neuron, excluding inputs)
    
    xIdx::Array{UnitRange{Int64},1} #indices of x values by level (includes input level)
    wIdx::Array{UnitRange{Int64},1} #indices of weights by level (excludes input level)
    bIdx::Array{UnitRange{Int64},1} #indices of biases by level (excludes input level)
    
    nWeights::Int64               #total number of weights
    nWeightsByLevel::Array{Int64} #number of weights by level
    
    da::Array{Float64}   #derivative of the activation function at each non-input neuron
    dCdb::Array{Float64} #derivative of the cost function wrt biases
    dCdw::Array{Float64} #derivative of the cost function wrt weights

    neuralNet() = new()

    function neuralNet(nNeurons::Array{Int64})
        nn = neuralNet();
        nn.nNeurons = nNeurons;
        
        nn.nLevels = length(nNeurons);

        #x value indices
        xIdx    = [0; cumsum(nNeurons)[1:end-1]].+1;
        nn.xIdx = [ xIdx[l]:(xIdx[l]+nn.nNeurons[l]-1) for l=1:nn.nLevels ];
        
        #weight indices
        nn.nWeightsByLevel = [ nNeurons[l]*nNeurons[l-1] for l=2:nn.nLevels ];
        nn.nWeights = sum( nn.nWeightsByLevel );
        wIdx = [0; cumsum(nn.nWeightsByLevel)[1:end-1]].+1;
        nn.wIdx = [ wIdx[l]:(wIdx[l]+nn.nWeightsByLevel[l]-1) for l=1:nn.nLevels-1 ];
        
        #biase indices
        nBiases = sum(nNeurons[2:end]);
        bIdx    = [0; cumsum(nNeurons[2:end-1])].+1;
        nn.bIdx = [ bIdx[l]:(bIdx[l]+nn.nNeurons[l+1]-1) for l=1:nn.nLevels-1 ];
        
        #allocate
        nn.x    = zeros(sum(nNeurons)); #activations & inputs
        nn.w    = zeros(nn.nWeights);   #weights
        nn.b    = zeros(nBiases);       #biases

        #allocate (use for back propagation)
        nn.da   = zeros(nBiases);      #derivative of activation function
        nn.dCdb = zeros(nBiases);      #derivative of cost wrt biases (delta)
        nn.dCdw = zeros(nn.nWeights);  #derivative of cost wrt weights
        
        return nn;
    end
end

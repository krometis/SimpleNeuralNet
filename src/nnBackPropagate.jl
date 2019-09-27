#back propagation for data y
#(see "Summary: the equations of backpropagation" at http://neuralnetworksanddeeplearning.com/chap2.html)
#requires that the function nnDCost be defined representing the derivative of the cost function
function nnBackPropagate(nn::neuralNet,datay::Array{Float64},nnDCost::Function=nnDCost)
    dC = nnDCost( datay, nn.x[nn.xIdx[nn.nLevels]] ); #error
    
    #output layer
    l = nn.nLevels;
    nn.dCdb[nn.bIdx[l-1]] = dC .* nn.da[nn.bIdx[l-1]];
    nn.dCdw[nn.wIdx[l-1]] = (nn.dCdb[nn.bIdx[l-1]] * nn.x[nn.xIdx[l-1]]')[:];
    
    #other layers
    for l=nn.nLevels-1:-1:2
        Wt = reshape(nn.w[nn.wIdx[l]],nn.nNeurons[l+1],nn.nNeurons[l]); #assumes that we've stored W' (W row-major)
        
        #compute derivative of cost wrt biases (delta)
        nn.dCdb[nn.bIdx[l-1]] = nn.da[nn.bIdx[l-1]] .* (Wt'*nn.dCdb[nn.bIdx[l]]);
        
        #compute derivative of cost wrt weights
        nn.dCdw[nn.wIdx[l-1]] = (nn.dCdb[nn.bIdx[l-1]] * nn.x[nn.xIdx[l-1]]')[:];
    end
end

#compute the output of the neural net for input x0
#requires that the functions nnAct() and nnDAct() be defined, 
#representing the activation function and its derivative, respectively
function nnForward(nn::neuralNet,x0::Array{Float64},nnAct::Function=nnAct,nnDAct::Function=nnDAct)
    #define inputs
    nn.x[nn.xIdx[1]] = x0;
    #loop over levels
    for l=1:nn.nLevels-1
        Wt = reshape(nn.w[nn.wIdx[l]],nn.nNeurons[l+1],nn.nNeurons[l]); #assumes that we've stored W' (row-major)
        z  = Wt*nn.x[nn.xIdx[l]] + nn.b[nn.bIdx[l]];
        nn.x[nn.xIdx[l+1]] = nnAct(z);
        nn.da[nn.bIdx[l]] = nnDAct(z); #save derivatives for later
    end
    
    return nn.x[nn.xIdx[nn.nLevels]]; #return NN output in case that's helpful
end

#train the neural net for dataset (datax,datay)
#requires that the function nnDCost be defined representing the derivative of the cost function
function nnTrain(nn::neuralNet,datax,datay,epochs::Int64,learnRate::Float64;batchSize=size(datax,2),verbose=false)
    #get indices of the data to be used
    (batchSize == size(datax,2)) ? (dIdx=1:batchSize) : (dIdx=sample(1:size(datax,2),batchSize,replace=false));
    
    #loop over epochs
    for ep=1:epochs
        #verbose && @printf("starting epoch %3d...\n",ep);
        
        #these vectors will store the running sums
        c  = 0.0;
        dw = zeros(length(nn.dCdw));
        db = zeros(length(nn.dCdb));

        #loop over training data
        for d=dIdx
            #pick out the dataset
            dx = datax[:,d];
            dy = datay[:,d];
            
            #forward run
            y = nnForward(nn,dx);
            
            #add costs to running sum
            c += nnCost(dy,y);

            #back propagate
            nnBackPropagate(nn,dy);
            
            #add computed gradients to running sum
            dw += nn.dCdw;
            db += nn.dCdb;
        end
        
        #shift by mean * learning rate
        nn.w -= learnRate*(dw ./ batchSize);
        nn.b -= learnRate*(db ./ batchSize);

        #verbose && println("finished epoch $ep.");
        verbose && @printf("finished epoch %3d. cost = %10.8f\n",ep,c/batchSize);
    end

end

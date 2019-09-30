using Statistics
using LinearAlgebra
using Printf

#function nnComputeCosts(nn,datax,datay;print=true,dIdx=1:size(datax,2))
#  c = [ nnCost(datay[:,d],nnForward(nn,datax[:,d])) for d=dIdx ];
#
#  #Print costs
#  if print
#    #for d=1:length(c)
#    #    @printf("dataset %3d: cost=%10.8f\n",d,c[d]);
#    #end
#    @printf("  mean costs: %10.8f\n",mean(c));
#    @printf("median costs: %10.8f\n",median(c));
#    @printf("2-norm costs: %10.8f\n",norm(c,2)/length(c));
#  end
#
#  return c;
#end
#In this version we get the neural net and input data
function nnComputeCosts(nn,datax,datay;print=true,dIdx=1:size(datax,2))
  outputs = nnComputeOutputs(nn,datax;dIdx=dIdx);
  #c = [ nnCost(datay[:,d],outputs[:,d]) for d=dIdx ];

  ##Print costs
  #if print
  #  #for d=1:length(c)
  #  #    @printf("dataset %3d: cost=%10.8f\n",d,c[d]);
  #  #end
  #  @printf("  mean costs: %10.8f\n",mean(c));
  #  @printf("median costs: %10.8f\n",median(c));
  #  @printf("2-norm costs: %10.8f\n",norm(c,2)/length(c));
  #end

  #return c;
  return nnComputeCosts(outputs,datay;print=print,dIdx=dIdx);
end

#In this version we just get the outputs
function nnComputeCosts(outputs,datay;print=true,dIdx=1:size(datay,2))
  c = [ nnCost(datay[:,d],outputs[:,d]) for d=dIdx ];

  #Print costs
  if print
    #for d=1:length(c)
    #    @printf("dataset %3d: cost=%10.8f\n",d,c[d]);
    #end
    @printf("  mean costs: %10.8f\n",mean(c));
    @printf("median costs: %10.8f\n",median(c));
    @printf("2-norm costs: %10.8f\n",norm(c,2)/length(c));
  end

  return c;
end


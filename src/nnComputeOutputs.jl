function nnComputeOutputs(nn,datax;dIdx=1:size(datax,2))
  out = zeros(nn.nNeurons[end],length(dIdx));
  for d=dIdx
    out[:,d] = nnForward(nn,datax[:,d]);
  end

  return out;
end


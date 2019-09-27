#Load a (trained) neural network from an HDF5 file
function nnLoad(fileName::String)
  #read from file
  f = h5open(fileName);
  nNeurons = read(f,"nNeurons");
  w        = read(f,"w");
  b        = read(f,"b");
  close(f);

  #build the neural net
  nn = neuralNet(nNeurons);

  #assign the saved weights and biases
  nn.w = w;
  nn.b = b;

  return nn;
end

#Save a (trained) neural network to an HDF5 file
function nnSave(nn::neuralNet,fileName::String)
  f = h5open(fileName,"w");
  write(f,"nNeurons",nn.nNeurons);
  write(f,"w",nn.w);
  write(f,"b",nn.b);
  close(f);
  println("Saved to: $fileName.");
end

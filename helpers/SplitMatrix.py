import numpy as np

def split_matrix(data, users, movies):
  # Convert the data set to the IxJ matrix  
  X = np.zeros((users, movies)) * np.nan

  for i in np.arange(len(data)):
    X[data[i,0]-1,data[i,1]-1] = data[i,2]
    
  return X

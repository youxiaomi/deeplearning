import torch

torch.set_printoptions(precision=4, sci_mode=False)

def prepare_multiclass_data(num_samples, num_features, num_classes):
  X = torch.randn(num_samples, num_features)
  y = torch.randint(0, num_classes, (num_samples,))
  return X,y


X_test , Y_test = prepare_multiclass_data(10,6,5)


def softmax(z):
  exp_z = torch.exp(z)
  return exp_z / torch.sum(exp_z)
  
probabilities =  softmax(X_test)
print(probabilities)

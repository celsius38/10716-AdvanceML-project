import math
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import GridInterpolationKernel as SKI
from gpytorch.kernels import SpectralMixtureKernel as SM
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
import matplotlib.image as img
from tqdm import tqdm
class GPModel(gpytorch.models.ExactGP):
    """
    A GP Model that performs exact gp inference
    :param train_x: torch.tensor
    :param train_y: torch.tensor
    :param likelihood: torch.tensor
    """
    def __init__(self, train_x, train_y, likelihood,kernel = 'rbf'):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        self.mean_module = gpytorch.means.ConstantMean()
        assert kernel in ['rbf','matern']
        if kernel =='rbf':
            self.covar_module = SKI(
                ScaleKernel(
                    RBFKernel(ard_num_dims=2)
                ), grid_size=grid_size, num_dims=2 
            )
        elif kernel == 'matern':
            self.covar_module = SKI(
                ScaleKernel(
                    MaternKernel()
                ), grid_size=grid_size, num_dims=2 
            )
            

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(model, likelihood, optimizer, training_iter:int, train_x: torch.tensor, train_y:torch.tensor):
    """ 
    Find optimal model hyperparameters
    """
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")
        optimizer.step()

def test(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    return observed_pred

def plot_img(img, *args, **kwargs):
    """
    plot an image without axis
    """
    plt.axis("off")
    plt.imshow(img, *args, **kwargs)
    plt.show()

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
    def __init__(self, train_x, train_y, likelihood,kernel = 'rbf',nu = 2.5):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel =='rbf':
            self.covar_module = SKI(
                ScaleKernel(
                    RBFKernel(ard_num_dims=2)
                ), grid_size=grid_size, num_dims=2 
            )
        elif kernel == 'matern':
            self.covar_module = SKI(
                ScaleKernel(
                    MaternKernel(nu)
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
def prep_training_img_ir(path: str, first_coord_crop:tuple, second_coord_crop:tuple) -> (torch.tensor, 
        torch.tensor, torch.tensor):
    """
    prepare the training image by occlude a patch of it
    :param path: path to the test image
    :param first_coord_crop: (start, end) the start and end of the first coordinate of the cropping area
    :param second_coord_crop: ... second coordinate ...
    : return : (1) train_x
               (2) train_y
               (3) test_x
               (4) occulated image
    """
    # prepare the training image
    training_img = img.imread(path)
    print(training_img.shape)

    training_img = training_img[::4,::4,:]
    print(training_img.shape)
    plot_img(training_img)
    plot_img(training_img[:,:,0], cmap=plt.cm.Reds_r)
    plot_img(training_img[:,:,1], cmap=plt.cm.Greens_r)
    plot_img(training_img[:,:,2], cmap=plt.cm.Blues_r)
    
    # occlude the original image
    crop_fs, crop_fe = first_coord_crop
    crop_ss, crop_se = second_coord_crop
    training_img[crop_fs:crop_fe, crop_ss:crop_se, :] = 0.99999
    plot_img(training_img)
    plot_img(training_img[:,:,0], cmap=plt.cm.Reds_r)
    plot_img(training_img[:,:,1], cmap=plt.cm.Greens_r)
    plot_img(training_img[:,:,2], cmap=plt.cm.Blues_r)
    
    # we store the x-coordinate in row-major order
    # train_x
    first_coord_cap, second_coord_cap, _ = training_img.shape
    train_x_first_coord = np.concatenate((
        np.repeat(np.arange(crop_fs), second_coord_cap),
        np.repeat(np.arange(crop_fs, crop_fe), second_coord_cap - (crop_se - crop_ss)),
        np.repeat(np.arange(crop_fe, first_coord_cap), second_coord_cap)
    ))
    train_x_second_coord = np.concatenate((
        np.tile(np.arange(second_coord_cap), crop_fs),
        np.tile(np.concatenate((np.arange(crop_ss),np.arange(crop_se,second_coord_cap))), crop_fe - crop_fs),
        np.tile(np.arange(second_coord_cap), first_coord_cap - crop_fe)
    ))
    train_x = np.stack((train_x_first_coord, train_x_second_coord)).T

    # train_y_rgb
    train_y_rgb = (training_img[train_x[:, 0], train_x[:, 1], 0], 
                   training_img[train_x[:, 0], train_x[:, 1], 1],
                   training_img[train_x[:, 0], train_x[:, 1], 2])

    # test_x
    test_x = np.meshgrid(np.arange(crop_fs, crop_fe), np.arange(crop_ss, crop_se))
    test_x = np.stack((test_x[0].T.ravel(), test_x[1].T.ravel())).T

    # convert x to float, and all to tensor
    train_y_rgb = list(map(lambda x: torch.tensor(x).float(), train_y_rgb))
    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    return train_x, train_y_rgb, test_x, training_img
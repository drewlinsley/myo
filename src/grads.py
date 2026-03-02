import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F


class VanillaGrad(object):
    """
    Implements the Vanilla Gradient visualization technique for neural networks.

    This class computes the gradients of the input with respect to the output
    of a specified class (or the highest scoring class if not specified).

    Attributes:
        pretrained_model (nn.Module): The pre-trained model to visualize.
        cuda (bool): Whether to use CUDA for computations.

    Methods:
        __call__(x, index=None): Compute the gradients for the input.
    """

    def __init__(self, pretrained_model, cuda=False):
        """
        Initialize the VanillaGrad object.

        Args:
            pretrained_model (nn.Module): The pre-trained model to visualize.
            cuda (bool): Whether to use CUDA for computations.
        """
        self.pretrained_model = pretrained_model
        self.cuda = cuda
        self.pretrained_model.eval()

    def __call__(self, x, index=None):
        """
        Compute the gradients for the input.

        Args:
            x (torch.Tensor): The input tensor.
            index (int, optional): The class index to compute gradients for.
                If None, the highest scoring class is used.

        Returns:
            numpy.ndarray: The computed gradients.
        """
        output = self.pretrained_model(x)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        if self.cuda:
            one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
        else:
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_variables=True)

        grad = x.grad.data.cpu().numpy()
        grad = grad[0, :, :, :]

        return grad


class SmoothGrad(VanillaGrad):
    """
    Implements the SmoothGrad visualization technique for neural networks.

    This class extends VanillaGrad by adding noise to the input and averaging
    the resulting gradients over multiple samples.

    Attributes:
        stdev_spread (float): The standard deviation of the noise relative to the input range.
        n_samples (int): The number of noisy samples to use.
        magnitude (bool): Whether to use the absolute value of gradients.
        cuda (bool): Whether to use CUDA for computations.
        use_decomp (bool): Whether to use decomposition for gradient computation.

    Methods:
        __call__(x, rotations, mu, sd, clfd, index=None, num_classes=8, thresh=2.):
            Compute the smoothed gradients for the input.
    """

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=20, magnitude=True, use_decomp=True):
        """
        Initialize the SmoothGrad object.

        Args:
            pretrained_model (nn.Module): The pre-trained model to visualize.
            cuda (bool): Whether to use CUDA for computations.
            stdev_spread (float): The standard deviation of the noise relative to the input range.
            n_samples (int): The number of noisy samples to use.
            magnitude (bool): Whether to use the absolute value of gradients.
            use_decomp (bool): Whether to use decomposition for gradient computation.
        """
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude
        self.cuda = cuda
        self.use_decomp = use_decomp

    def __call__(self, x, rotations, mu, sd, clfd, nmf_arg, index=None, num_classes=8, thresh=2.):
        """
        Compute the smoothed gradients for the input.

        Args:
            x (torch.Tensor): The input tensor.
            rotations (torch.Tensor): Rotation matrix for decomposition.
            mu (torch.Tensor): Mean for normalization.
            sd (torch.Tensor): Standard deviation for normalization.
            clfd (torch.Tensor): Not used in the current implementation.
            index (int, optional): The class index to compute gradients for.
                If None, the highest scoring class is used.
            num_classes (int): The number of classes in the model's output.
            thresh (float): Threshold for masking in decomposition mode.

        Returns:
            numpy.ndarray: The computed smoothed gradients.
        """
        x_cpu = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x_cpu) - np.min(x_cpu))
        total_gradients = np.zeros_like(x_cpu)
        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x_cpu.shape).astype(np.float32)
            x_plus_noise = x_cpu + noise
            if self.cuda:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).cuda(), requires_grad=True)
            else:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise), requires_grad=True)

            if self.use_decomp:
                _, _, latents = self.pretrained_model(x_plus_noise, return_features=True)
                latents = latents[:, :, :latents.shape[2] // 2]
                output = latents.mean((2, 3, 4))
                output = output.squeeze()
                output = (output - mu) / sd
                output = torch.matmul(output, rotations)
                output = -output  # Flip sign to focus on sparks
                mask = (output > thresh)
                loss = (output * mask).sum()
            else:
                output = self.pretrained_model(x_plus_noise)
                if index is None:
                    index = output.squeeze().argmax(1)

                one_hot = F.one_hot(index, num_classes).float()
                if self.cuda:
                    one_hot = Variable(one_hot, requires_grad=True)
                else:
                    one_hot = Variable(one_hot.cpu(), requires_grad=True)
                loss = torch.sum(one_hot * output.squeeze())

                if x_plus_noise.grad is not None:
                    x_plus_noise.grad.data.zero_()

            loss.backward()
            grad = x_plus_noise.grad.data.cpu().numpy()

            if self.magnitutde:
                total_gradients += np.abs(grad)  # use abs not squaring (grad * grad)
            else:
                total_gradients += grad

        avg_gradients = total_gradients / self.n_samples

        with torch.no_grad():
            _, _, latents = self.pretrained_model(x, return_features=True)
            latents = latents[:, :, :latents.shape[2] // 2].mean((2, 3, 4)).cpu().numpy()
            score = clfd.transform(latents)[:, nmf_arg]

        return avg_gradients, score

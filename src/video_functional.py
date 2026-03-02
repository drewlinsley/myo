import numbers
import torch
import cv2
import numpy as np
import PIL
from PIL import Image


def _is_tensor_clip(clip):
    """
    Check if the input is a 4-dimensional tensor clip.

    Args:
        clip (torch.Tensor): The input to be checked.

    Returns:
        bool: True if the input is a 4-dimensional tensor, False otherwise.
    """
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    """
    Crop a video clip to the specified dimensions.

    Args:
        clip (list): A list of frames (numpy arrays or PIL Images).
        min_h (int): The starting height for cropping.
        min_w (int): The starting width for cropping.
        h (int): The height of the crop.
        w (int): The width of the crop.

    Returns:
        list: A list of cropped frames.

    Raises:
        TypeError: If the input clip is not a list of numpy arrays or PIL Images.
    """
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def to_grayscale(img, num_output_channels=1):
    """
    Convert an image to grayscale.

    Args:
        img (PIL.Image.Image): Image to be converted to grayscale.
        num_output_channels (int): Number of channels in the output image (1 or 3).

    Returns:
        PIL.Image.Image: Grayscale version of the image.

    Raises:
        TypeError: If the input is not a PIL Image.
        ValueError: If num_output_channels is not 1 or 3.
    """
    if not isinstance(img,PIL.Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img

def resize_clip(clip, size, interpolation='bilinear'):
    """
    Resize a video clip to the specified size.

    Args:
        clip (list): A list of frames (numpy arrays or PIL Images).
        size (int or tuple): Desired output size. If size is an int, the smaller edge of the image will be matched
                             to this number maintaining the aspect ratio. If size is a tuple of (height, width),
                             the image will be resized to match the tuple.
        interpolation (str): Interpolation method to use. Either 'bilinear' or 'nearest'.

    Returns:
        list: A list of resized frames.

    Raises:
        TypeError: If the input clip is not a list of numpy arrays or PIL Images.
    """
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            # pil_inter = PIL.Image.NEAREST
            pil_inter = PIL.Image.BILINEAR
        else:
            # pil_inter = PIL.Image.BILINEAR
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    """
    Calculate the new height and width for resizing an image while maintaining aspect ratio.

    Args:
        im_h (int): Original image height.
        im_w (int): Original image width.
        size (int): Desired size for the smaller edge.

    Returns:
        tuple: A tuple containing the new height and width (oh, ow).
    """
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    """
    Normalize a tensor video clip by mean and standard deviation.

    Args:
        clip (torch.Tensor): Tensor video clip to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace (bool): Whether to perform the operation in-place.

    Returns:
        torch.Tensor: Normalized tensor video clip.

    Raises:
        TypeError: If the input clip is not a torch tensor clip.
    """
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

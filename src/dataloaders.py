import torch
import numpy as np
from torch.utils.data import Dataset
from src import video_functional as VF


class MSData_train(Dataset):
    """
    Dataset class for training multi-scale data.

    Args:
        bright (numpy.ndarray): Input image data.
        label (numpy.ndarray): Corresponding labels.
        mu (float): Mean value for normalization.
        std (float): Standard deviation for normalization.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, bright, label, mu, std, transform):
        label = label[..., None]

        # We are assuming one label for all images
        label = label[None].repeat(len(bright), 0)

        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()  # Normalize with dataset mean/variance
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())  # Normalize to [0, 1] for pretrained

        # Reduce mu/std to one channel
        bright = bright[..., None]
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std  # Normalize to pretrained weights

        self.all = np.concatenate((bright.astype(np.float32), label), -1)
        self.all = self.all.transpose(0, 3, 1, 2)
        self.all = torch.from_numpy(self.all)
        self.transform = transform

    def __len__(self):
        return len(self.all)

    def __getitem__(self, idx):
        sample = self.all[idx]
        if self.transform:
            sample = self.transform(sample)
        m = sample[[0]].float()
        l = sample[1].long()
        return m, l


class MSData_test(Dataset):

    def __init__(self, bright, mu, std, transform):
        """
        Initialize the MSData_test dataset.

        Args:
            bright (numpy.ndarray): Input image data.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # self.original = torch.from_numpy(bright.copy()).long()
        self.original = bright.copy().astype(np.float32)
        self.original = (self.original - self.original.min()) / (self.original.max() - self.original.min())
        self.original = self.original * 255
        self.original = self.original[..., None]

        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()  # Normalize with dataset mean/variance
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())  # Normalize to [0, 1] for pretrained

        # Reduce mu/std to one channel
        bright = bright[..., None]
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std  # Normalize to pretrained weights

        self.all = np.concatenate((bright.astype(np.float32), self.original), -1)
        self.all = self.all.transpose(0, 3, 1, 2)
        self.all = self.all.transpose(0, 3, 1, 2)
        self.all = torch.from_numpy(self.all)
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.all)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (m, l) where m is the transformed image and l is the label.
        """
        sample = self.all[idx]
        if self.transform:
            sample = self.transform(sample)
        m = sample[[0]].float()
        l = sample[1].long()
        return m, l


class IntraData_train(Dataset):

    def __init__(self, bright, label, mu, std, dims, transform=None, time=32):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # label = label[None, None].repeat(len(bright), 1, 1, 1)  # Torch
        label = label[None, ..., None].repeat(len(bright), 0)  # Numpy
        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()  # Normalize with dataset mean/variance
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())  # Normalize to [0, 1] for pretrained

        # Reduce mu/std to one channel
        bright = bright[..., None]
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std  # Normalize to pretrained weights
        # self.all = torch.concat((bright, label), 1)
        self.all = np.concatenate((bright, label), -1).astype(np.float32)
        # self.all = self.all.transpose(1, 0, 2, 3)[:, None]
        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.all) - self.time - 1  # Adjust for timesteps

    def __getitem__(self, idx):

        index = torch.arange(idx, idx + self.time)
        sample = self.all[index]
        if self.transform:
            sample = self.transform(sample)
            if self.dims == "BCDHW":
                sample = sample.permute(3, 0, 1, 2)
        m = sample[[0]]  # Images
        l = sample[1].long()  # Labels
        l = l[[-1]]  # Fit to final timestep
        return m, l


class IntraData_test(Dataset):

    def __init__(self, bright, mu, std, dims, transform=None, time=32):
        """
        Args:
            bright (numpy.ndarray): Input image data.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
        """
        # label = label[None, None].repeat(len(bright), 1, 1, 1)  # Torch
        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()  # Normalize with dataset mean/variance
        self.original = bright.copy()
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())  # Normalize to [0, 1] for pretrained

        # Reduce mu/std to one channel
        bright = bright[..., None].astype(np.float32)
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std  # Normalize to pretrained weights
        # self.all = torch.concat((bright, label), 1)
        self.all = bright
        # self.all = self.all.transpose(1, 0, 2, 3)[:, None]
        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.all) - self.time - 1  # Adjust for timesteps

    def __getitem__(self, idx):

        index = torch.arange(idx, idx + self.time)
        sample = self.all[index]
        o = self.original[index]
        if self.transform:
            sample = self.transform(sample)
            if self.dims == "BCDHW":
                sample = sample.permute(3, 0, 1, 2)
        m = sample[[0]].float()  # Images
        return m, o


class IntraData_train(Dataset):

    def __init__(self, bright, label, mu, std, dims, transform=None, time=32):
        """
        Dataset class for training intra-patient data.

        Args:
            bright (numpy.ndarray): Input image data.
            label (numpy.ndarray): Corresponding labels.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
        """
        # label = label[None, None].repeat(len(bright), 1, 1, 1)  # Torch
        label = label[None, ..., None].repeat(len(bright), 0)  # Numpy
        data_mu, data_std = bright.ravel().mean(), bright.ravel().std()  # Normalize with dataset mean/variance
        bright = (bright - data_mu) / data_std
        bright = (bright - bright.min()) / (bright.max() - bright.min())  # Normalize to [0, 1] for pretrained

        # Reduce mu/std to one channel
        bright = bright[..., None]
        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        bright = (bright - mu) / std  # Normalize to pretrained weights
        # self.all = torch.concat((bright, label), 1)
        self.all = np.concatenate((bright, label), -1).astype(np.float32)
        # self.all = self.all.transpose(1, 0, 2, 3)[:, None]
        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.all) - self.time - 1  # Adjust for timesteps

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (m, l) where m is the image data and l is the label.
        """
        index = torch.arange(idx, idx + self.time)
        sample = self.all[index]
        if self.transform:
            sample = self.transform(sample)
            if self.dims == "BCDHW":
                sample = sample.permute(3, 0, 1, 2)
        m = sample[[0]]  # Images
        l = sample[1].long()  # Labels
        l = l[[-1]]  # Fit to final timestep
        return m, l


class Classification_train(Dataset):

    def __init__(self, files, labels, mu, std, dims, pre_HW, transform=None, time=32):
        """
        Dataset class for training classification tasks.

        Args:
            files (list): List of file paths to load data from.
            labels (list): Corresponding labels for each file.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            pre_HW (tuple): Target height and width for resizing.
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
        """
        self.files = files
        self.labels = np.asarray(labels)

        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        self.mu = mu
        self.std = std
        self.pre_HW = pre_HW

        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx]).astype(np.float32)
        data = data[..., [0]]  # Reduce to 1 channel data
        label = self.labels[idx]

        # Normalize to min/max
        data = (data - data.min()) / (data.max() - data.min())

        # Get random time chunk
        diff = len(data) - self.time
        if diff < 0:
            data = np.pad(data, ((0, np.abs(diff)), (0, 0), (0, 0), (0, 0)))
        elif diff > 0:
            start = np.random.randint(diff)
            data = data[start: start + self.time]

        # Zscore for TIMM
        data = (data - self.mu) / self.std

        # Resize to pre_HW
        rdata = VF.resize_clip(data, self.pre_HW)
        rdata = np.asarray(rdata)[..., None]

        # Transform
        if self.transform:
           rdata = self.transform(rdata)

        if self.dims == "BCDHW":
            rdata = rdata.permute(3, 0, 1, 2)

        rdata = rdata.float()
        return rdata, label


class Classification_test(Dataset):

    def __init__(self, files, labels, mu, std, dims, pre_HW, transform=None, time=32):
        """
        Dataset class for testing classification tasks.

        Args:
            files (list): List of file paths to load data from.
            labels (list): Corresponding labels for each file.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            pre_HW (tuple): Target height and width for resizing.
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
        """
        self.files = files
        self.labels = np.asarray(labels)

        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        self.mu = mu
        self.std = std
        self.pre_HW = pre_HW

        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fl = self.files[idx]
        data = np.load(fl).astype(np.float32)
        data = data[..., [0]]  # Reduce to 1 channel data
        label = self.labels[idx]

        # Normalize to min/max
        data = (data - data.min()) / (data.max() - data.min())

        # Get random time chunk
        diff = len(data) - self.time
        if diff < 0:
            data = np.pad(data, ((0, np.abs(diff)), (0, 0), (0, 0), (0, 0)))
        elif diff > 0:
            start = np.random.randint(diff)
            data = data[start: start + self.time]

        # Zscore for TIMM
        data = (data - self.mu) / self.std

        # Resize to pre_HW
        rdata = VF.resize_clip(data, self.pre_HW)
        rdata = np.asarray(rdata)[..., None]

        # Transform
        if self.transform:
           rdata = self.transform(rdata)

        if self.dims == "BCDHW":
            rdata = rdata.permute(3, 0, 1, 2)

        rdata = rdata.float()
        return rdata, label, fl


class Classification_test_whole(Dataset):

    def __init__(self, files, wholes, masks, labels, mu, std, dims, pre_HW, transform=None, time=32):
        """
        Dataset class for testing classification tasks with whole images and masks.

        Args:
            files (list): List of file paths to load data from.
            wholes (list): List of file paths for whole images.
            masks (list): List of file paths for masks.
            labels (list): Corresponding labels for each file.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            pre_HW (tuple): Target height and width for resizing.
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
        """
        self.files = files
        self.wholes = wholes
        self.masks = masks
        self.labels = np.asarray(labels)

        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        self.mu = mu
        self.std = std
        self.pre_HW = pre_HW

        self.time = time
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, interpolation="nearest"):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
            interpolation (str, optional): Interpolation method for resizing. Defaults to "nearest".

        Returns:
            tuple: (rdata, wrdata, mask, label, fl) where rdata is the processed image data,
                   wrdata is the processed whole image data, mask is the processed mask,
                   label is the corresponding label, and fl is the file path.
        """
        fl = self.files[idx]
        wl = self.wholes[idx]
        msk = self.masks[idx]
        data = np.load(fl).astype(np.float32)
        data = data[..., [0]]  # Reduce to 1 channel data
        data_wl = np.load(wl)[..., [0]].astype(np.float32)
        mask = np.load(msk).astype(np.float32)
        label = self.labels[idx]

        # Check if mask is in "first" or "all" conditions
        if len(mask.shape) == 2:
            mask = mask[None, ..., None].repeat(len(data), 0)

        # Normalize to min/max
        data = (data - data.min()) / (data.max() - data.min())

        # Normalize a copy of whole to min/max
        ndata_wl = (data_wl - data_wl.min()) / (data_wl.max() - data_wl.min())

        # Get random time chunk
        diff = len(data) - self.time
        if diff < 0:
            data = np.pad(data, ((0, np.abs(diff)), (0, 0), (0, 0), (0, 0)))
            data_wl = np.pad(data_wl, ((0, np.abs(diff)), (0, 0), (0, 0), (0, 0)))
            ndata_wl = np.pad(ndata_wl, ((0, np.abs(diff)), (0, 0), (0, 0), (0, 0)))
        elif diff > 0:
            start = np.random.randint(diff)
            data = data[start: start + self.time]
            data_wl = data_wl[start: start + self.time]
            ndata_wl = ndata_wl[start: start + self.time]

        mask_diff = len(mask) - self.time
        if mask_diff < 0:
            mask = np.pad(mask, ((0, np.abs(mask_diff)), (0, 0), (0, 0), (0, 0)))
        elif mask_diff > 0:
            mask = mask[start: start + self.time]

        # Zscore for TIMM
        data = (data - self.mu) / self.std
        ndata_wl = (ndata_wl - self.mu) / self.std

        # Resize to pre_HW
        rdata = VF.resize_clip(data, self.pre_HW, interpolation=interpolation)
        rdata = np.asarray(rdata)[..., None]
        wrdata = VF.resize_clip(data_wl, self.pre_HW, interpolation=interpolation)
        wrdata = np.asarray(wrdata)[..., None]
        nwrdata = VF.resize_clip(ndata_wl, self.pre_HW, interpolation=interpolation)
        nwrdata = np.asarray(nwrdata)[..., None]
        mask = VF.resize_clip(mask, self.pre_HW, interpolation=interpolation)
        mask = np.asarray(mask)[..., None]

        # Transform
        if self.transform:
            rdata = self.transform(rdata)
            wrdata = self.transform(wrdata)
            nwrdata = self.transform(nwrdata)
            mask = self.transform(mask)

        if self.dims == "BCDHW":
            rdata = rdata.permute(3, 0, 1, 2)
            wrdata = wrdata.permute(3, 0, 1, 2)
            nwrdata = nwrdata.permute(3, 0, 1, 2)

        rdata = rdata.float()
        wrdata = wrdata.float()
        nwrdata = nwrdata.float()
        return rdata, nwrdata, wrdata, mask, label, fl


class NextFrame_train(Dataset):

    def __init__(self, files, labels, mu, std, dims, pre_HW, transform=None, time=32, offset=32):
        """
        Dataset class for training next frame prediction tasks.

        Args:
            files (list): List of file paths to load data from.
            labels (list): Corresponding labels for each file.
            mu (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            dims (str): Dimension order (e.g., 'BCDHW').
            pre_HW (tuple): Target height and width for resizing.
            transform (callable, optional): Optional transform to be applied on a sample.
            time (int, optional): Number of time steps. Defaults to 32.
            offset (int, optional): Offset for next frame prediction. Defaults to 32.
        """
        self.files = files
        self.labels = np.asarray(labels)

        mu = np.asarray(mu[0]).reshape(1, 1, 1, 1)
        std = np.asarray(std[0]).reshape(1, 1, 1, 1)
        self.mu = mu
        self.std = std
        self.pre_HW = pre_HW

        self.time = time
        self.offset = offset
        self.transform = transform
        self.dims = dims

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx]).astype(np.float32)
        data = data[..., [0]]  # Reduce to 1 channel data
        label = self.labels[idx]

        # Normalize to min/max
        data = (data - data.min()) / (data.max() - data.min())

        # Zscore for TIMM
        data = (data - self.mu) / self.std

        # Get random time chunk
        start = np.random.randint(len(data) - self.time - self.offset)
        data = data[start: start + self.time + self.offset]

        # Resize to pre_HW
        rdata = VF.resize_clip(data, self.pre_HW)
        rdata = np.asarray(rdata)[..., None]

        # Transform
        if self.transform:
           rdata = self.transform(rdata)

        if self.dims == "BCDHW":
            rdata = rdata.permute(3, 0, 1, 2)

        rdata = rdata.float()
        X = rdata[:, :-self.offset]
        y = rdata[:, -self.offset:]
        return X, y


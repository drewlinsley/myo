import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class HeatMap:
    """
    A class for creating and manipulating heatmaps overlaid on images.

    Attributes:
        image (np.ndarray): The base image on which the heatmap will be overlaid.
        heat_map (np.ndarray): The heatmap data.
    """

    def __init__(self, image, heat_map, gaussian_std=10):
        """
        Initialize the HeatMap object.

        Args:
            image (np.ndarray): The base image as a numpy array.
            heat_map (np.ndarray): The heatmap data.
            gaussian_std (int, optional): Standard deviation for Gaussian smoothing. Defaults to 10.

        Raises:
            NotImplementedError: If the image is not a numpy array.
        """
        if isinstance(image, np.ndarray):
            self.image = image
        else:
            raise NotImplementedError("Only numpy array images are currently supported.")
        
        self.heat_map = heat_map

    def plot(self, transparency=0.5, color_map='Reds', show_axis=False, show_original=False, show_colorbar=False, width_pad=0, thresh=0.05):
        """
        Plot the heatmap overlaid on the image.

        Args:
            transparency (float, optional): Alpha value for heatmap transparency. Defaults to 0.5.
            color_map (str, optional): Colormap for the heatmap. Defaults to 'Reds'.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            show_original (bool, optional): Whether to show the original image alongside the heatmap. Defaults to False.
            show_colorbar (bool, optional): Whether to show the colorbar. Defaults to False.
            width_pad (int, optional): Padding between subplots. Defaults to 0.
            thresh (float, optional): Threshold for masking the heatmap. Defaults to 0.05.
        """
        #If show_original is True, then subplot first figure as orginal image
        #Set x,y to let the heatmap plot in the second subfigure, 
        #otherwise heatmap will plot in the first sub figure
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x,y=2,2
        else:
            x,y=1,1
        
        #Plot the heatmap
        plt.subplot(1,x,y)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image, cmap="Greys_r")
        masked = np.ma.masked_where(self.heat_map < thresh, self.heat_map)
        plt.imshow(masked, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.show()

    def return_img(self, transparency=0.5, color_map='Reds', show_axis=False, show_original=False, show_colorbar=False, width_pad=0, thresh=0.05):
        """
        Generate and return the heatmap image as a numpy array.

        Args:
            transparency (float, optional): Alpha value for heatmap transparency. Defaults to 0.5.
            color_map (str, optional): Colormap for the heatmap. Defaults to 'Reds'.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            show_original (bool, optional): Whether to show the original image alongside the heatmap. Defaults to False.
            show_colorbar (bool, optional): Whether to show the colorbar. Defaults to False.
            width_pad (int, optional): Padding between subplots. Defaults to 0.
            thresh (float, optional): Threshold for masking the heatmap. Defaults to 0.05.

        Returns:
            np.ndarray: The generated heatmap image as a numpy array.
        """
        #Plot the heatmap
        f, ax = plt.subplots(1, 1)
        canvas = FigureCanvasAgg(f)
        plt.imshow(self.image, cmap="Greys_r")
        masked = np.ma.masked_where(self.heat_map < thresh, self.heat_map)
        plt.imshow(masked, alpha=transparency, cmap=color_map)
        ax.margins(0)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        plt.close("all")
        return X

    def save(self, filename, format='png', save_path=os.getcwd(), transparency=0.7, color_map='bwr', width_pad=-10, show_axis=False, show_original=False, show_colorbar=False, **kwargs):
        """
        Save the heatmap image to a file.

        Args:
            filename (str): Name of the file to save.
            format (str, optional): File format. Defaults to 'png'.
            save_path (str, optional): Directory to save the file. Defaults to current working directory.
            transparency (float, optional): Alpha value for heatmap transparency. Defaults to 0.7.
            color_map (str, optional): Colormap for the heatmap. Defaults to 'bwr'.
            width_pad (int, optional): Padding between subplots. Defaults to -10.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            show_original (bool, optional): Whether to show the original image alongside the heatmap. Defaults to False.
            show_colorbar (bool, optional): Whether to show the colorbar. Defaults to False.
            **kwargs: Additional keyword arguments to pass to plt.savefig().

        Prints:
            A message confirming the file has been saved successfully.
        """
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x,y=2,2
        else:
            x,y=1,1
        
        #Plot the heatmap
        plt.subplot(1,x,y)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image)
        plt.imshow(self.heat_map/255, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.savefig(os.path.join(save_path,filename+'.'+format), 
                    format=format, 
                    bbox_inches='tight',
                    pad_inches = 0, **kwargs)
        print('{}.{} has been successfully saved to {}'.format(filename, format, save_path))

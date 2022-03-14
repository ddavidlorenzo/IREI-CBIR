from json import load
import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def remove_filename_from_path(out_filename:str, path_standard_format=False):
    """Attempts to remove filename from the provided path.

    :param out_filename: Filepath.
    :type out_filename: str
    :param path_standard_format: Indicates whether the path follows the standard
     format (backslash separator) or the slash separator, defaults to False.
    :type path_standard_format: bool, optional
    :return: The directory excluding the filename.
    :rtype: str
    """
    if path_standard_format:
        out_filename = out_filename.replace('\\', '/')
    return (out_filename if '/' not in out_filename else
        out_filename.replace(out_filename.split('/')[-1], ''))

def makedir(path:str, remove_filename=False, recursive=True, exist_ok=True):
    """Creates directory from path if not exists.
    
    :param path: Path of the directory to be created.
    :type path: str
    :param remove_filename: If set to True, it attempts to remove the filename from
     the path, defaults to False
    :type remove_filename: bool, optional
    :param recursive: Creates directories recursively (i.e., create necessary 
     subdirectories if necessary), defaults to True
    :type recursive: bool, optional
    :param exist_ok: is set to False, arises an error if `path` directory exists,
     defaults to True
    :type exist_ok: bool, optional
    """
    if '/' in path or '\\' in path:
        path = path if not remove_filename else remove_filename_from_path(path)
        Path(path).mkdir(parents=recursive, exist_ok=exist_ok)

def load_serialized_data(filename:str, return_dict_values=False):
    """Utility to load serialized data (and other optional stored values)
    from disk using *pickle*.
    
    :param str filename: Filename of the file to be loaded.
    :param return_dict_values: If set to True, returns the values just the values
     of the dictionary containing all stored data, defaults to False.
    :type return_dict_values: bool, optional
    :return: Loaded data
    """
    # Load embeddings and other stored information from disk
    with open(filename, "rb") as fIn:
        stored_data = pickle.load(fIn)
    return stored_data.values() if return_dict_values else stored_data

def store_serialized_data(corpus_embeddings, out_filename='colour_embeddings.pkl',
        protocol=pickle.HIGHEST_PROTOCOL, **kwargs):
    """Utility to dump embeddings (and other optional values indicated in the 
    keyword arguments) to disk using *pickle*.

    :param corpus_embeddings: Tensor type data structure containing the dominant
     colours of the dataset
    :param out_filename: Path for the output file, defaults to 'colour_embeddings.pkl'.
    :type out_filename: str, optional
    :param protocol: Protocol used for *pickle*, defaults to `pickle.HIGHEST_PROTOCOL`.
    """
    # Create directory if it does not exist.
    makedir(out_filename, remove_filename=True)
    with open(out_filename, "wb") as fOut:
        pickle.dump(corpus_embeddings, fOut, protocol=protocol)
  
def scan_files(fdir, file_extension="jpg"):
    """ Scan files recursively from a directory with a specific file extension.

    :param fdir: Source directory.
    :param out_filename: File extension, defaults to 'jpg'.
    :type out_filename: str, optional
    """
    return np.array([os.path.join(dirname, filename)
            for dirname, _, filenames in os.walk(fdir) 
                for filename in filenames 
                    if filename.endswith(f".{file_extension}")
           ])

def load_image(func):
    def inner(img, **kwargs):
        if isinstance(img, np.str_) or isinstance(img, str):
            img = cv2.imread(img)
        return func(img, **kwargs)
    return inner

def _get_fig_layout(n_graphs, gs_x=5, gs_y=5, ncols=None):
    """Get optimal plot layout for `n_graphs` subplots.

    :param n_graphs: Number of subplots
    :type n_graphs: int
    :param gs_x: Size of the subplots in the x axis, defaults to 5
    :type gs_x: int, optional
    :param gs_y: Size of the subplots in the y axis, defaults to 5
    :type gs_y: int, optional
    :param ncols: Number of columns of the plot, defaults to None
    :type ncols: int, optional
    :return: Figure layout
    """
    # If the number of columns is not fixed
    if not ncols:
        # obtain a square layout
        sqrt = math.sqrt(n_graphs)
        ncols = sqrt + 1 if sqrt - int(sqrt) > 0 else sqrt
        nrows = ncols
    else: 
        nrows = n_graphs//ncols if n_graphs%ncols == 0 \
                else n_graphs//ncols +1 
        
    return (nrows, ncols), plt.figure(figsize=(gs_x*ncols, gs_y*nrows))

def plot_img_grid(data, col_space_conv=None, **kwargs):
    """Plot grid of data

    :param data: iterable of filepaths to images or plots
    :type data: iterable
    """
    grid_dim, fig = _get_fig_layout(len(data), **kwargs)
    for i, img in enumerate(data):
        # If `img` is a path to an image, then read it from disk.
        if isinstance(img, np.str_) or isinstance(img, str):
            img = cv2.imread(img)
        num = 1 + i
        ax = fig.add_subplot(*grid_dim, num)
        ax.axis('off')
        if col_space_conv:
            ax.imshow(cv2.cvtColor(img, col_space_conv))
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

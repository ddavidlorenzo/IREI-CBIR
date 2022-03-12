from json import load
import cv2
import glob2
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

def load_embeddings(filename:str, return_dict_values=False):
    """Utility to load embeddings (and other optional stored values) from disk
    using *pickle*.
    
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

def store_embeddings(corpus_embeddings, out_filename='embeddings.pkl',
        protocol=pickle.HIGHEST_PROTOCOL, **kwargs):
    """Utility to dump embeddings (and other optional values indicated in the 
    keyword arguments) to disk using *pickle*.

    :param corpus_embeddings: Tensor type data structure containing the embeddings
     for the corpus.
    :param out_filename: Path for the output file, defaults to 'embeddings.pkl'.
    :type out_filename: str, optional
    :param protocol: Protocol used for *pickle*, defaults to `pickle.HIGHEST_PROTOCOL`.
    """
    # Create directory if it does not exist.
    makedir(out_filename, remove_filename=True)
    with open(out_filename, "wb") as fOut:
        pickle.dump(corpus_embeddings, fOut, protocol=protocol)

            
def load_files_from_dirs_to_mm(*pattern_dirs, file_extension="jpg"):
    return np.array([f for directory in pattern_dirs for f in glob2.glob(directory + "\\*." + file_extension)])

def rgb_hist(image, size_hist=256, showRes=True):
    bgr_planes = cv2.split(image)
    histRange = (0, size_hist) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [size_hist], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [size_hist], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [size_hist], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/size_hist ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    for i in range(1, size_hist):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)

    if showRes:
        cv2.imshow('Source image', image)
        cv2.imshow('calcHist Demo', histImage)
        cv2.waitKey()
    return histImage

def dominant_colours(img, n_colours=5, show_palette=False):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2LAB)
    pixels = np.float32(img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colours, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    indices = np.argsort(counts)[::-1]   
    freqs = [counts[indices]/float(counts.sum())]

    if show_palette:
        cum_freq = np.cumsum(np.hstack([0], cum_freq))
        rows = np.int_(img.shape[0]*cum_freq)
        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        cv2.imshow('Dominant colors', dom_patch)
        cv2.waitKey()
    
    return palette, freqs

def serialize_dominant_colours(dataset, filename="colour_embeddings.pkl", **kwargs):
    dic = {file:dominant_colours(file, **kwargs) for file in dataset}
    store_embeddings(dic, filename)
    return dic

def resize_img(img, width=250, height=None):
    if not (width or height):
        raise ValueError('Either the new width or height must be specified.')
    if width and height:
        pass 
    elif width:
        resize_prop = width/img.shape[1]
        height = int(img.shape[0]*resize_prop)
    elif height:
        resize_prop = height/img.shape[0]
        width = int(img.shape[1]*resize_prop)

    # resize image
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)



def colour_img_score(colour, palette):
    return min([np.linalg.norm(colour-c) for c in palette])

def search_by_colour(colour, dataset, serial_dc=None, topk=10):
    if serial_dc:
        dc = load_embeddings(serial_dc)
        # Override existing definition of dominant colours extraction.
        dominant_colours = lambda img: dc[img]

    scores = np.argsort([colour_img_score(colour, dominant_colours(img)[0]) for img in dataset])
    return dataset[scores[:topk]]




dir_origin = 'pokemon_dataset\\'
serial_path = 'colour_embeddings.pkl'

dc = load_embeddings(serial_path, return_dict_values=False)
# for dirname, _, filenames in os.walk(dir_origin):
#     pokemon = dirname.split('/')
#     print(pokemon)
dataset = np.array([os.path.join(dirname, filename)
                        for dirname, _, filenames in os.walk(dir_origin) 
                            for filename in filenames 
                                if filename.endswith(".jpg")
                    ]
                   )

# serialize_dominant_colours(dataset)

search_c = np.array([232, 118, 217])

results = search_by_colour(search_c, dataset, serial_dc=serial_path)

for i, img in enumerate(results):
    img = cv2.imread(img)
    cv2.imshow(f'top_{i}', img)
    cv2.waitKey()


# rimg = resize_img(src)
# palette, freqs = dominant_colours(rimg)
# search_c = np.array([240, 204, 25])

# euc_distances = np.argsort([np.linalg.norm(search_c-c) for c in palette])

 

# print(palette[euc_distances[0]])


# rgb_hist(rimg)
# print(rimg.shape)
# cv2.imshow('calcHist Demo', rimg)
# cv2.waitKey()


# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
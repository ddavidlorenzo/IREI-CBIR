import cv2
import os
import numpy as np
from utils import *

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

def dominant_colours(img, n_colours=5, show_palette=False, 
                     kmeans_max_iter=200, kmeans_eps=0.1,
                     kmeans_init=cv2.KMEANS_RANDOM_CENTERS,
                     kmeans_max_attempts=10):
    """Compute the `n_colours` most representative colours of
    an image located in the path `img`.

    :param img: Path to an image
    :type img: str
    :param n_colours: Number of representative colours to get 
     from the image, defaults to 5
    :type n_colours: int, optional
    :param show_palette: show the resultant colour palette, 
     defaults to False
    :type show_palette: bool, optional
    :param kmeans_max_iter: Maximum number of iterations for k-means,
     defaults to 200
    :type kmeans_max_iter: int, optional
    :param kmeans_eps: Epsilon value for k-means, defaults to 0.1
    :type kmeans_eps: float, optional
    :param kmeans_init: K-means nitialization method, defaults 
     to cv2.KMEANS_RANDOM_CENTERS
    :type kmeans_init: int, optional
    :param kmeans_max_attempts: Maximum number of attempts of the k-means
     algorithm, defaults to 10
    :type kmeans_max_attempts: int, optional
    :return: Collection of dominant colours and their frequence
     of appearance.
    """
    # Read image usingk l*a*b colour space.
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2LAB)
    # Flatten pixel values
    pixels = np.float32(img.reshape(-1, 3))
    
    # k-means termination criteria. By default, max number of iterations = 200;  epsilon = 0.1.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_max_iter, kmeans_eps)
    # By default, initialize seeds at random.
    flags = kmeans_init

    # Perform k-means clustering with the previous setup.
    _, labels, palette = cv2.kmeans(pixels, n_colours, None, criteria, kmeans_max_attempts, flags)
    # Compute the cardinality of the constituent colour subgroups.
    _, counts = np.unique(labels, return_counts=True)

    # Sort them in decreasing order respect to their recurrence
    indices = np.argsort(counts)[::-1]
    # Compute the relative frequencies of each cluster   
    freqs = np.array([counts[indices]/float(counts.sum())])

    # Plot the result
    if show_palette:
        # compute cumulative frequencies
        cum_freq = np.cumsum(np.hstack(([0], np.squeeze(freqs))))
        # height of the colours correspond to their weight,
        # encoded in their frequence. 
        rows = np.int_(img.shape[0]*cum_freq)
        dom_img = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            # fill each region of pixels with the corresponding colour.
            dom_img[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        # plot the original image, along with that of its dominant colours in
        # the RGB colour space.
        plot_img_grid([img, dom_img], col_space_conv=cv2.COLOR_LAB2RGB, ncols=1)

    # Return the collection of dominant colours and their frequence
    return palette, freqs

def serialize_dominant_colours(dataset, filename="colour_serial.pkl", **kwargs):
    """Serialize the computed dominant colours of the images in a dataset.

    :param dataset: dataset of images.
    :type dataset: iterable collection of images.
    :param filename: output filepath, defaults to "colour_serial.pkl"
    :type filename: str, optional
    :return: collection of dominant colours and their frequence 
    """
    dic = {file:dominant_colours(file, **kwargs) for file in dataset}
    store_serialized_data(dic, filename)
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
    """Returns the minimum euclidean distance between the target colour
    and the dominant colours of an image, encoded in `palette`

    :param colour: _description_
    :type colour: _type_
    :param palette: Collection of dominant colours
    :type palette: Iterable
    :return: Distance to the semantically closest colour to the target colour.
    :rtype: float
    """
    return min([np.linalg.norm(colour-c) for c in palette])

def search_by_colour(colour, dataset, serial_dc=None, topk=10):
    """Search images in `dataset` matching with similar colours to `colour`.

    :param colour: query colour
    :type colour: array like (e.g., `np.array`)
    :param dataset: collection of filepaths to images
    :type dataset: iterable of str objects
    :param serial_dc: precomputed collection of dominant colours for each object in
     the dataset, defaults to None
    :type serial_dc: dict, optional
    :param topk: top k images to retrieve, defaults to 10
    :type topk: int, optional
    :return: top k most relevant images found
    :rtype: list
    """
    if serial_dc:
        dc = load_serialized_data(serial_dc)
        # Override existing definition of dominant colours extraction.
        scores = np.argsort([colour_img_score(colour, dc[img][0]) for img in dataset])
    else:
        scores = np.argsort([colour_img_score(colour, dominant_colours(img)[0]) for img in dataset])
    return dataset[scores[:topk]]



# b= np.array([[[233, 192, 129]]], dtype='uint8')
# color = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)


base_dir = 'pokemon_dataset\\'
serial_path = 'colour_serial.pkl'

dataset = scan_files(base_dir)

# serialize_dominant_colours(dataset)

# Yellow
search_c = np.array([232, 118, 217])

# Purple
# search_c = np.array([168, 158, 88])

# # Orange
# search_c = np.array([168, 167, 199])

# # Pink
# search_c = np.array([226, 141, 132])

# # Blue
# search_c = np.array([191, 118, 101])

SAMPLE_IMAGE = "pokemon_dataset\\Abra\\10a9f06ec6524c66b779ea80354f8519.jpg"
dominant_colours(SAMPLE_IMAGE, show_palette=True)

# img = cv2.cvtColor(cv2.imread("colours\\lila.jpg"), cv2.COLOR_BGR2LAB)

results = search_by_colour(search_c, dataset, serial_dc=serial_path)

plot_img_grid(results)

# rgb_hist(rimg)
# print(rimg.shape)
# cv2.imshow('calcHist Demo', rimg)
# cv2.waitKey()


# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
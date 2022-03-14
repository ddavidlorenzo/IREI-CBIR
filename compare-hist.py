import numpy as np
import cv2
from utils import *

BASE_DIR = 'pokemon_dataset\\'
SAMPLE_IMAGE = "colours\\10a9f06ec6524c66b779ea80354f8519.jpg"
SERIAL_PATH = 'hist_serial.pkl'
# initialize OpenCV methods for histogram comparison
COMPARE_METHODS = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi-square": cv2.HISTCMP_CHISQR,
    "intersection": cv2.HISTCMP_INTERSECT,
    "hellinger": cv2.HISTCMP_BHATTACHARYYA
}

def serialize_hist(dataset, filename="hist_serial.pkl", **kwargs):
    """Serialize the computed dominant colours of the images in a dataset.

    :param dataset: dataset of images.
    :type dataset: iterable collection of images.
    :param filename: output filepath, defaults to "colour_serial.pkl"
    :type filename: str, optional
    :return: collection of dominant colours and their frequence 
    """
    dic = {file:calc_hist(file, **kwargs) for file in dataset}
    store_serialized_data(dic, filename)
    return dic


def img_grid_even(img, hgrid=3, wgrid=3):
    h, w, _ = img.shape
    return [img[int(h/hgrid)*j:int(h/hgrid)*(j+1), 
                int(w/wgrid)*i:int(w/wgrid)*(i+1)] 
            for j in range(hgrid) for i in range(wgrid)]

@load_image
def calc_hist(img, n_channels=3, per_channel_bins = 8, per_channel_range=(0,256)):
    """ Compute a 3D RGB color histogram from the image, using 
    `per_channel_bins` bins per channel.

    :param img: _description_
    :type img: _type_
    :param n_channels: _description_
    :type n_channels: _type_
    :param per_channel_bins: _description_, defaults to 8
    :type per_channel_bins: int, optional
    :param per_channel_range: _description_, defaults to (0,256)
    :type per_channel_range: tuple, optional
    :return: _description_
    :rtype: _type_
    """
    hist = cv2.calcHist([img], list(range(n_channels)), None, [per_channel_bins]*n_channels,
                        [*per_channel_range]*n_channels)

    return cv2.normalize(hist, hist).flatten()
    
# divide en n 
# para cada n -> calc_hist

def grid_wise_compare_hist(grid1, grid2, **kwargs):
    return


def search_by_colour_hist(img, dataset, compare_method, serial_hist=None, topk=10):
    if compare_method not in COMPARE_METHODS:
        raise ValueError(f'Invalid compare method. Try again with one of correlation {list(compare_method.keys())}.')
    # The histogram of the image query should be computed just once.
    hist_img = calc_hist(img)
    if serial_hist:
        sh = load_serialized_data(serial_hist)
        # Override existing definition of colour histogram calculus.
        scores = np.argsort([cv2.compareHist(hist_img, sh[dimg], COMPARE_METHODS[compare_method]) for dimg in dataset])
    else:
        scores = np.argsort([cv2.compareHist(hist_img, calc_hist(dimg), COMPARE_METHODS[compare_method]) for dimg in dataset])
    if compare_method=="correlation" or compare_method=="intersection":
        scores = scores[::-1]
    return dataset[scores[:topk]]


dataset = scan_files(BASE_DIR)

# serialize_hist(dataset)
histograms = load_serialized_data(SERIAL_PATH)

image=SAMPLE_IMAGE

results = search_by_colour_hist(SAMPLE_IMAGE, dataset, "hellinger", serial_hist=SERIAL_PATH)

plot_img_grid(results)

# print(dataset[:3])
# a = type(dataset[1])



# # extract a 3D RGB color histogram from the image,
# # using 8 bins per channel, normalize, and update
# # the index
# n_channels = 3 # RGB

# sample_hist = calc_hist(image)

# index = {k:calc_hist(k) for k in dataset[:4]}
# images = {k:cv2.imread(k) for k in dataset[:4]}



# # loop over the comparison methods
# for methodName, method in COMPARE_METHODS.items():
#     # initialize the results dictionary and the sort
#     # direction
#     results = {}
#     reverse = False
#     # if we are using the correlation or intersection
#     # method, then sort the results in reverse order
#     if methodName in ("Correlation", "Intersection"):
#         reverse = True

#     # loop over the index
#     for (k, hist) in index.items():
#         # compute the distance between the two histograms
#         # using the method and update the results dictionary
#         d = cv2.compareHist(sample_hist, hist, method)
#         results[k] = d
#     # sort the results
#     results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)


#     # show the query image
#     fig = plt.figure("Query")
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(cv2.imread(SAMPLE_IMAGE))
#     plt.axis("off")
#     # initialize the results figure
#     fig = plt.figure("Results: %s" % (methodName))
#     fig.suptitle(methodName, fontsize = 20)
#     # loop over the results
#     for (i, (v, k)) in enumerate(results):
#         # show the result
#         ax = fig.add_subplot(1, len(images), i + 1)
#         ax.set_title("%s: %.2f" % (k, v))
#         plt.imshow(images[k])
#         plt.axis("off")
#     # show the OpenCV methods
#     plt.show()
import cv2
import glob2
import numpy as np
import matplotlib.pyplot as plt

def load_files_from_dirs_to_mm(*pattern_dirs, file_extension="jpg"):
    # return [cv2.imread(f) for f in [glob.glob(directory + "*." + file_extension) for directory in pattern_dirs]]
    # return [f for directory in pattern_dirs for f in glob.glob(directory + "*." + file_extension)]
    return {f:cv2.imread(f) for directory in pattern_dirs for f in glob2.glob(directory + "*." + file_extension)}

a = load_files_from_dirs_to_mm('./')

src = list(a.values())[1]

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

def dominant_colours(img, n_colours=10, show_palette=False):
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

rimg = resize_img(src)
palette, freqs = dominant_colours(rimg)
search_c = np.array([240, 204, 25])

euc_distances = np.argsort([np.linalg.norm(search_c-c) for c in palette])

    

print(palette[euc_distances[0]])


rgb_hist(rimg)
print(rimg.shape)
cv2.imshow('calcHist Demo', rimg)
cv2.waitKey()


# https://pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
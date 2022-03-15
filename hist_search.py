import numpy as np
import cv2
import utils
import os
from img_search import ImgSearch

class HistogramSearch(ImgSearch):

    COMPARE_METHODS = {
        "correlation": cv2.HISTCMP_CORREL,
        "chi-square": cv2.HISTCMP_CHISQR,
        "intersection": cv2.HISTCMP_INTERSECT,
        "hellinger": cv2.HISTCMP_BHATTACHARYYA
    }

    __serial_hist = None
    __serial_grid_hist = None
    
    def __init__(self,
                dataset,
                serial_hist_path=None,
                serial_grid_hist_path=None
                ):
        """_summary_

        :param dataset: dataset of images.
        :type dataset: iterable collection of images.
        :param serial_hist_path: _description_, defaults to None
        :type serial_hist_path: _type_, optional
        :param serial_grid_hist_path: _description_, defaults to None
        :type serial_grid_hist_path: _type_, optional
        """
        super().__init__(dataset)
        if serial_hist_path: 
            self.serial_hist = serial_hist_path
        if serial_grid_hist_path:
            self.serial_grid_hist = serial_grid_hist_path


    def serialize(self, filename="serial\\hist_serial.pkl", grid=False, **kwargs):
        """Serialize the histogram of colours of the images in a dataset.

        :param filename: output filepath, defaults to "serial\\colour_serial.pkl"
        :type filename: str, optional
        :param grid: whether to compute histograms by regions (in a grid-like fashion),
        defaults to False
        :type grid: bool, optional
        :return: collection of dominant colours and their frequence 
        """
        dic = {file:self.calc_hist(file, **kwargs) for file in self.dataset} if grid \
            else {file:[self.calc_hist(dimg, **kwargs) for dimg in self.img_grid(file, **kwargs)]
                     for file in self.dataset}

        utils.store_serialized_data(dic, filename)
        return dic

    @ImgSearch.load_image
    def img_grid(self, img, hgrid=5, wgrid=5):
        h, w, _ = img.shape
        return [img[int(h/hgrid)*j:int(h/hgrid)*(j+1), 
                    int(w/wgrid)*i:int(w/wgrid)*(i+1)] 
                for j in range(hgrid) for i in range(wgrid)]

    @ImgSearch.load_image
    def calc_hist(self, img, n_channels=3, per_channel_bins = 8, per_channel_range=(0,256)):
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
    
    def check_valid_compare_method(func):
        def inner(self, img, compare_method, **args):
            if compare_method not in self.COMPARE_METHODS:
                raise ValueError(f'Invalid compare method. Try again with one of correlation {list(self.COMPARE_METHODS.keys())}.')
            return func(self, img, compare_method, **args)
        return inner

    @check_valid_compare_method
    def smart_search(self, img, compare_method, topk=10, **kwargs):
        # Histograms of the image query should be computed just once.
        hist_grid_img = [self.calc_hist(simg) for simg in self.img_grid(img, **kwargs)]

        if self.serial_grid_hist:
            scores = np.argsort(
                    [
                        sum(
                            [cv2.compareHist(himg, gdimg,
                                             self.COMPARE_METHODS[compare_method]) 
                                for himg, gdimg in zip(hist_grid_img, self.serial_grid_hist[dimg])]
                        ) for dimg in self.dataset
                    ]
                )
        else:    
            scores = np.argsort(
                    [
                        sum(
                            [cv2.compareHist(himg, self.calc_hist(gdimg),
                                             self.COMPARE_METHODS[compare_method]) 
                                for himg, gdimg in zip(hist_grid_img, self.img_grid(dimg, **kwargs))]
                        ) for dimg in self.dataset
                    ]
                )

        if compare_method=="correlation" or compare_method=="intersection":
            scores = scores[::-1]
        return self.dataset[scores[:topk]]  

    @check_valid_compare_method
    def search(self, img, compare_method, topk=10, **kwargs):
        # The histogram of the image query should be computed just once.
        hist_img = self.calc_hist(img, **kwargs)
        if self.serial_hist:
            scores = np.argsort([cv2.compareHist(hist_img, 
                                                 self.serial_hist[dimg], 
                                                 self.COMPARE_METHODS[compare_method]
                                                ) 
                                    for dimg in self.dataset
                                ])
        else:
            scores = np.argsort([cv2.compareHist(hist_img, 
                                                 self.calc_hist(dimg),
                                                 self.COMPARE_METHODS[compare_method])
                                    for dimg in self.dataset
                                ])

        if compare_method=="correlation" or compare_method=="intersection":
            scores = scores[::-1]
        return self.dataset[scores[:topk]]

    @property
    def serial_hist(self):
        return self.__serial_hist

    @serial_hist.setter
    def serial_hist(self, path):
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_hist = utils.load_serialized_data(path)
        else:
            print(f"File {path} does not exists. Creating a new index in that path.")
            self.__serial_hist = self.serialize(path)
            return self.__serial_hist

    @property
    def serial_grid_hist(self):
        return self.__serial_grid_hist

    @serial_grid_hist.setter
    def serial_grid_hist(self, path, hgrid=5, wgrid=5):
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_grid_hist = utils.load_serialized_data(path)
        else:
            print(f"File '{path}' does not exists. Creating a new index in that path with params 'hgrid'={hgrid}, 'wgrid'={wgrid}.")
            self.__serial_hist = self.serialize(path)
            return self.__serial_grid_hist



if __name__ == "__main__":
    BASE_DIR = 'pokemon_dataset\\'
    SAMPLE_IMAGE = "colours\\10a9f06ec6524c66b779ea80354f8519.jpg"
    SAMPLE_IMAGE = "pokemon_dataset\\Aerodactyl\\00000093.jpg"
    #Este es bueno
    SAMPLE_IMAGE = "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg"

    SERIAL_PATH = 'serial\\hist_serial.pkl'
    HGRID=5
    WGRID=5
    SERIAL_GRID_PATH = f'serial\\hist_serial_w{HGRID}_h{WGRID}.pkl'
    # initialize OpenCV methods for histogram comparison
    histogram_search = HistogramSearch(BASE_DIR, serial_hist_path=SERIAL_PATH, serial_grid_hist_path=SERIAL_GRID_PATH)

    results = histogram_search.smart_search(SAMPLE_IMAGE, "hellinger")
    utils.plot_img_grid(results)
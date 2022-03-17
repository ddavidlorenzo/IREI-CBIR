from typing import Callable, Iterable, List, Union
import numpy as np
import cv2
import utils
import os
from img_search import ImgSearch

class HistogramSearch(ImgSearch):
    """This class implements a histogram search IR method, which allows 
    for features like standard and smart histogram based content 
    retrieval.
    """
    # Histogram comparison methods
    COMPARE_METHODS = {
        "corr": cv2.HISTCMP_CORREL,
        "chisq": cv2.HISTCMP_CHISQR,
        "intersect": cv2.HISTCMP_INTERSECT,
        "hellinger": cv2.HISTCMP_BHATTACHARYYA
    }

    # Precomputed computed histograms
    __serial_hist = None
    # Precomputed grid histograms.
    __serial_grid_hist = None
    
    def __init__(self,
                dataset:Union[Iterable[np.array],str],
                serial_hist_path:str=None,
                serial_grid_hist_path:str=None
                ) -> None:
        """Constructor for a `HistogramSearch` instance.

        :param dataset: dataset of images or path to the directory
         containg input samples.
        :type dataset: Union[Iterable[np.array],str]
        :param serial_hist_path: Path to serialize computed histograms, defaults to None
        :type serial_hist_path: str, optional
        :param serial_grid_hist_path: Path to serialize computed grid 
         histograms, defaults to None
        :type serial_grid_hist_path: str, optional
        """
        super().__init__(dataset)
        if serial_hist_path: 
            self.serial_hist = serial_hist_path
        if serial_grid_hist_path:
            self.serial_grid_hist = serial_grid_hist_path


    def serialize(self, filename:str="serial\\hist_serial.pkl", grid:bool=False, **kwargs):
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
    def img_grid(self, img:Union[np.array,str], hgrid:int=5, wgrid:int=5)->List[np.array]:
        """Divide an image in equally-sized fragments, conforming a grid with 
        `hgrid` columns and `wgrid` rows.

        :param img: image or path to target image.
        :type img: Union[np.array,str]
        :param hgrid: number of columns to compute the image grid, defaults to 5
        :type hgrid: int, optional
        :param wgrid: number of rows to compute the image grid, defaults to 5
        :type wgrid: int, optional
        :return: grid of subimages.
        :rtype: List[np.array]
        """
        h, w, _ = img.shape
        return [img[int(h/hgrid)*j:int(h/hgrid)*(j+1), 
                    int(w/wgrid)*i:int(w/wgrid)*(i+1)] 
                for j in range(hgrid) for i in range(wgrid)]

    @ImgSearch.load_image
    def calc_hist(self, 
                  img:Union[np.array,str], 
                  n_channels:int = 3, 
                  per_channel_bins:int = 8,
                  per_channel_range:int = (0,256)
                 ) -> np.array:
        """ Compute a 3D RGB color histogram from the image, using 
        `per_channel_bins` bins per channel.

        :param img: image or path to target image.
        :type img: Union[np.array,str]
        :param n_channels: number of channels of the image, defaults to 3
        :type n_channels: int
        :param per_channel_bins: number of bins per each used dimension,
         defaults to 8
        :type per_channel_bins: int, optional
        :param per_channel_range: range of values to be measured per each 
         dimension, defaults to (0,256)
        :type per_channel_range: tuple, optional
        :return: histogram of the image
        :rtype: np.array
        """
        hist = cv2.calcHist([img], list(range(n_channels)), None, [per_channel_bins]*n_channels,
                            [*per_channel_range]*n_channels)

        return cv2.normalize(hist, hist).flatten()
    
    def check_valid_compare_method(func:Callable)->Callable:
        """Checks whether the chosen comparison method is valid.

        :param func: callable function
        :type func: Callable
        """
        def inner(self, img, compare_method, **args):
            if compare_method not in self.COMPARE_METHODS:
                raise ValueError(f'Invalid compare method. Try again with one of {list(self.COMPARE_METHODS.keys())}.')
            return func(self, img, compare_method, **args)
        return inner

    @check_valid_compare_method
    def smart_search(self, 
                     img:Union[np.array,str], 
                     compare_method:str, 
                     topk:int=10, 
                     **kwargs
                    ) -> List[np.array]:
        """Performs smart histogram search.

        :param img: image or path to target image.
        :type img: Union[np.array,str]
        :param compare_method: comparison method
        :type compare_method: str
        :param topk: top k images to retrieve, defaults to 10
        :type topk: int, optional
        :return: top k most relevant images
        :rtype: List[np.array]
        """
        # Histograms of the image query should be computed just once.
        hist_grid_img = [self.calc_hist(simg) for simg in self.img_grid(img, **kwargs)]

        # Speed-up search by using cached histogram data.
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

        # High values obtain for the correlation and intersection method
        # indicate higher degree of similarity.
        if compare_method=="corr" or compare_method=="intersect":
            scores = scores[::-1]

        return self.dataset[scores[:topk]]  

    @check_valid_compare_method
    def search(self, 
               img:Union[np.array, str],
               compare_method:str,
               topk:int=10,
               **kwargs
               ) -> List[np.array]:
        """Performs basic histogram search. On the regular, less accurate than
         smart histogram search, albeit noticeably faster for large datasets.

        :param img: image or path to target image.
        :type img: Union[np.array,str]
        :param compare_method: comparison method
        :type compare_method: str
        :param topk: top k images to retrieve, defaults to 10
        :type topk: int, optional
        :return: top k most relevant images
        :rtype: List[np.array]
        """
        # The histogram of the image query should be computed just once.
        hist_img = self.calc_hist(img, **kwargs)

        # Speed-up search by using cached histogram data.
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

        # High values obtain for the correlation and intersection method
        # indicate higher degree of similarity.
        if compare_method=="correlation" or compare_method=="intersection":
            scores = scores[::-1]

        return self.dataset[scores[:topk]]

    @property
    def serial_hist(self) -> dict:
        """Getter method for `serial_hist`

        :return: precomputed histograms.
        :rtype: dict
        """
        return self.__serial_hist

    @serial_hist.setter
    def serial_hist(self, path:str) -> None:
        """Set or update the filepath to load (store) serialized histograms.

        :param path: filepath to load (store) serialized histograms.
        :type path: str
        """
        # If the index exists, load it off disk.
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_hist = utils.load_serialized_data(path)
        # otherwise, dump data to filepath `path`
        else:
            print(f"File {path} does not exists. Creating a new index in that path.")
            self.__serial_hist = self.serialize(path)

    @property
    def serial_grid_hist(self)->dict:
        """Getter method for `serial_grid_hist`

        :return: precomputed grid histograms.
        :rtype: dict
        """
        return self.__serial_grid_hist

    @serial_grid_hist.setter
    def serial_grid_hist(self, path:str, hgrid:int=5, wgrid:int=5)->None:
        """Set or update the filepath to load (store) smart serialized histograms.

        :param path: filepath to load (store) serialized smart histograms.
        :type path: str
        :param hgrid: number of columns to compute the image grid, defaults to 5
        :type hgrid: int, optional
        :param wgrid: number of rows to compute the image grid, defaults to 5
        :type wgrid: int, optional
        """
        # If the index exists, load it off disk.
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_grid_hist = utils.load_serialized_data(path)
        # otherwise, dump data to filepath `path`
        else:
            print(f"File '{path}' does not exists. Creating a new index in that path with params 'hgrid'={hgrid}, 'wgrid'={wgrid}.")
            self.__serial_hist = self.serialize(path)



# Execution example
if __name__ == "__main__":
    BASE_DIR = 'pokemon_dataset\\'
    SAMPLE_IMAGE = "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg"
    SERIAL_PATH = 'serial\\hist_serial.pkl'
    HGRID=5
    WGRID=5
    SERIAL_GRID_PATH = f'serial\\hist_serial_w{WGRID}_h{HGRID}.pkl'
   
    histogram_search = HistogramSearch(BASE_DIR, serial_hist_path=SERIAL_PATH, serial_grid_hist_path=SERIAL_GRID_PATH)

    results = histogram_search.smart_search(SAMPLE_IMAGE, "hellinger")
    utils.plot_img_grid(results)
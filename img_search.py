from typing import Callable, Iterable, Union
import utils
import numpy as np
import cv2

class ImgSearch(object):
    """Generic class for image search methods."""

    # Collection of images
    __dataset = None

    def __init__(self, dataset:Union[Iterable[np.ndarray],str]) -> None:
        self.dataset = dataset

    def serialize(self):
        raise NotImplementedError("Method not implemented")

    def search(self):
        raise NotImplementedError("Method not implemented")

    def load_image(func:Callable) -> Callable:
        """Load an image from disk, if required.

        :param func: callable function
        :type func: Callable
        """
        def inner(self, img, **kwargs):
            if isinstance(img, np.str_) or isinstance(img, str):
                img = cv2.imread(img)
            return func(self, img, **kwargs)
        return inner

    @property
    def dataset(self)->Iterable[np.ndarray]:
        """Getter method for `dataset`

        :return: collection of images.
        :rtype: Iterable[np.ndarray]
        """
        return self.__dataset

    @dataset.setter
    def dataset(self, data:Union[Iterable[np.ndarray],str]) -> None:
        """ Set or update the collection of images to use.

        :param dataset: dataset of images or path to the directory
         containg input samples.
        :type dataset: Union[Iterable[np.ndarray],str]
        """
        if type(data) == str:
            print(f'Loading data from path "{data}"')
            self.__dataset = utils.scan_files(data)
        else:
            self.__dataset = data
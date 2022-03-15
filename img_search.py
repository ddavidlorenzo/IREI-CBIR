import utils
import numpy as np
import cv2

class ImgSearch(object):
    __dataset = None

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def serialize(self):
        raise NotImplementedError("Method not implemented")

    def search(self):
        raise NotImplementedError("Method not implemented")

    def load_image(func):
        def inner(self, img, **kwargs):
            if isinstance(img, np.str_) or isinstance(img, str):
                img = cv2.imread(img)
            return func(self, img, **kwargs)
        return inner

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, data):
        if type(data) == str:
            print(f'Loading data from path "{data}"')
            self.__dataset = utils.scan_files(data)
        else:
            self.__dataset = data
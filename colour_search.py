from typing import Iterable, List, Tuple, Union
import cv2
import os
import numpy as np
import utils
from img_search import ImgSearch

class ColourSearch(ImgSearch):
    """This class implements a colour search IR method."""

    # Precomputed dominant colours of the constituent images.
    __serial_colour = None
    
    def __init__(self,
                dataset: Union[Iterable[np.array],str],
                serial_colour_path:str=None,
                ) -> None:
        """Constructor for a `HistogramSearch` instance.

        :param dataset: dataset of images or path to the directory
         containg input samples.
        :type dataset: iterable collection of images.
        :param serial_colour_path: path to serialize the computed dominant 
         colours, defaults to None.
        :type serial_colour_path: str, optional
        """
        super().__init__(dataset)
        if serial_colour_path: 
            self.serial_colour = serial_colour_path

    @property
    def serial_colour(self) -> dict:
        """Getter method for `serial_colour`

        :return: precomputed dominant colours of all images in the dataset.
        :rtype: dict
        """
        return self.__serial_colour

    @serial_colour.setter
    def serial_colour(self, path:str) -> None:
        """Set or update the filepath to load (store) the dominant colours computed.

        :param path: filepath to load (store) the dominant colours precomputed (computed).
        :type path: str
        """
        # If the index exists, load it off disk.
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_colour = utils.load_serialized_data(path)
        # otherwise, dump data to filepath `path`
        else:
            print(f"File {path} does not exists. Creating a new index in that path.")
            self.__serial_colour = self.serialize(path)

    def dominant_colours(self, 
                         img:str,
                         n_colours:int=5,
                         show_palette:bool=False, 
                         kmeans_max_iter:int=200, 
                         kmeans_eps:float=0.1,
                         kmeans_init:int=cv2.KMEANS_RANDOM_CENTERS,
                         kmeans_max_attempts:int=10
                        ) -> Tuple[Iterable[np.array], Iterable[np.array]]:
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
        :rtype: Tuple[Iterable[np.array], Iterable[np.array]]
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
            utils.plot_img_grid([img, dom_img], col_space_conv=cv2.COLOR_LAB2RGB, ncols=1)

        # Return the collection of dominant colours and their frequence
        return palette, freqs

    def serialize(self, filename:str="serial\\colour_serial.pkl", **kwargs) -> Iterable[np.array]:
        """Serialize the computed dominant colours of the images in a dataset.

        :param filename: output filepath, defaults to "serial\\colour_serial.pkl"
        :type filename: str, optional
        :return: collection of dominant colours and their frequence 
        """
        dic = {file:self.dominant_colours(file, **kwargs) for file in self.dataset}
        utils.store_serialized_data(dic, filename)
        return dic

    def score_colour_img(self, colour:np.array, palette:Iterable[np.array]) -> float:
        """Returns the minimum euclidean distance between the target colour
        and the dominant colours of an image, encoded in `palette`

        :param colour: target colour
        :type colour: np.array
        :param palette: Collection of dominant colours
        :type palette: Iterable[np.array]
        :return: Distance to the semantically closest colour to the target colour.
        :rtype: float
        """
        return min([np.linalg.norm(colour-c) for c in palette])

    def search(self, colour:np.array, topk=10)->List[np.array]:
        """Search images in `dataset` matching with similar colours to `colour`.

        :param colour: query colour
        :type colour: array like (e.g., `np.array`)
        :param topk: top k images to retrieve, defaults to 10
        :type topk: int, optional
        :return: top k most relevant images found
        :rtype: List[np.array]
        """
        if self.serial_colour:
            scores = np.argsort([self.score_colour_img(colour, self.serial_colour[img][0]) for img in self.dataset])
        else:
            scores = np.argsort([self.score_colour_img(colour, self.dominant_colours(img)[0]) for img in self.dataset])
        return self.dataset[scores[:topk]]


# Execution example
if __name__ == "__main__":
    BASE_DIR = 'pokemon_dataset\\'
    
    # b= np.array([[[233, 192, 129]]], dtype='uint8')
    # color = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)

    # Yellow
    COLOUR = np.array([232, 118, 217])
    SERIAL_PATH = 'serial\\colour_serial.pkl'

    colour_search = ColourSearch(BASE_DIR, serial_colour_path=SERIAL_PATH)

    results = colour_search.search(COLOUR)
    utils.plot_img_grid(results)

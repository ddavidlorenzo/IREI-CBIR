import cv2
import os
import numpy as np
import utils
from img_search import ImgSearch

class ColourSearch(ImgSearch):
    __serial_colour = None
    
    def __init__(self,
                dataset,
                serial_colour_path=None,
                ):
        """_summary_

        :param dataset: _description_
        :type dataset: _type_
        :param serial_colour_path: precomputed collection of dominant colours for each object in
        the dataset, defaults to None
        :type serial_colour_path: dict, optional
        """
        super().__init__(dataset)
        if serial_colour_path: 
            self.serial_colour = serial_colour_path

    @property
    def serial_colour(self):
        return self.__serial_colour

    @serial_colour.setter
    def serial_colour(self, path):
        if os.path.exists(path):
            print(f'Loading serialized data from path "{path}"')
            self.__serial_colour = utils.load_serialized_data(path)
        else:
            print(f"File {path} does not exists. Creating a new index in that path.")
            self.__serial_colour = self.serialize(path)
            return self.__serial_colour

    def dominant_colours(self, img, n_colours=5, show_palette=False, 
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
            utils.plot_img_grid([img, dom_img], col_space_conv=cv2.COLOR_LAB2RGB, ncols=1)

        # Return the collection of dominant colours and their frequence
        return palette, freqs

    def serialize(self, filename="serial\\colour_serial.pkl", **kwargs):
        """Serialize the computed dominant colours of the images in a dataset.

        :param filename: output filepath, defaults to "serial\\colour_serial.pkl"
        :type filename: str, optional
        :return: collection of dominant colours and their frequence 
        """
        dic = {file:self.dominant_colours(file, **kwargs) for file in self.dataset}
        utils.store_serialized_data(dic, filename)
        return dic

    def score_colour_img(self, colour, palette):
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

    def search(self, colour, topk=10):
        """Search images in `dataset` matching with similar colours to `colour`.

        :param colour: query colour
        :type colour: array like (e.g., `np.array`)
        :param topk: top k images to retrieve, defaults to 10
        :type topk: int, optional
        :return: top k most relevant images found
        :rtype: list
        """
        if self.serial_colour:
            scores = np.argsort([self.score_colour_img(colour, self.serial_colour[img][0]) for img in self.dataset])
        else:
            scores = np.argsort([self.score_colour_img(colour, self.dominant_colours(img)[0]) for img in self.dataset])
        return self.dataset[scores[:topk]]


if __name__ == "__main__":
    BASE_DIR = 'pokemon_dataset\\'

    # b= np.array([[[233, 192, 129]]], dtype='uint8')
    # color = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)

    # Yellow
    COLOUR = np.array([232, 118, 217])

    ## Purple
    # search_c = np.array([168, 158, 88])

    # Orange
    # search_c = np.array([168, 167, 199])

    # # Pink
    # search_c = np.array([226, 141, 132])

    # # Blue
    # search_c = np.array([191, 118, 101])

    SERIAL_PATH = 'serial\\colour_serial.pkl'

    # initialize OpenCV methods for histogram comparison
    colour_search = ColourSearch(BASE_DIR, serial_colour_path=SERIAL_PATH)

    results = colour_search.search(COLOUR)

    utils.plot_img_grid(results)

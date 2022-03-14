import cv2
from utils import *

SAMPLE_IMAGE = "pokemon_dataset\\Abra\\e297786c64574fbbb264dc6274aa5864.jpg"
img = cv2.imread(SAMPLE_IMAGE)
h, w, _= img.shape


def img_grid_even(img, hgrid=3, wgrid=3):
    return [img[int(h/hgrid)*j:int(h/hgrid)*(j+1), 
                int(w/wgrid)*i:int(w/wgrid)*(i+1)] 
            for j in range(hgrid) for i in range(wgrid)]

wgrid=3
plot_img_grid(img_grid_even(img), ncols=wgrid)


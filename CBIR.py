import argparse
from colour_search import ColourSearch
from hist_search import HistogramSearch
import numpy as np
from utils import plot_img_grid

def get_serial_path(search_method, serialize, wgrid=None, hgrid=None):        
    return dict(
        smarthist = f'serial\\hist_serial_w{wgrid}_h{hgrid}.pkl',
        hist = "serial\\hist_serial.pkl",
        col = "serial\\colour_serial.pkl"
    )[search_method] if serialize else None

def load_input_colour(colour):
    if colour!= "custom":
        return dict(
            blue = np.array([191, 118, 101]),
            orange = np.array([168, 167, 199]),
            pink = np.array([226, 141, 132]),
            purple = np.array([168, 158, 88]),
            yellow = np.array([232, 118, 217]),
        )[colour]
    else:
        return np.array([float(i) for i in input("Input space-separated values of a colour in l*a*b space, e.g., '191 118.5 101': ").strip().split(' ')])

# Starting point for CBIR.
if __name__ == '__main__':
    # Argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", help="Path of the folder containing the input files "
                + "to be fed into the CBIR.")
    parser.add_argument("search", default="dtr", choices=["smarthist", "hist", "col"],
                help='Type of search to conduct. Can be one of: "hist" '
                    + 'for histogram search, "smarthist" for smart histogram search, '
                    + 'or "col" for colour search')
    parser.add_argument("-c", "--colour", choices=["blue", "orange", "pink", "purple", "yellow", "custom"],
                help='Input colour to CBIR for colour search.')
    parser.add_argument("-i", "--image",
                help="Path to input image to CBIR for histogram search")
    parser.add_argument("-t", "--topk", type=int, default=10,
                help="Top k images to retrieve")
    parser.add_argument("-hg", "--hgrid", type=int, default=5,
                help="Number of columns to compute image grids, both in image query and dataset samples")
    parser.add_argument("-wg", "--wgrid", type=int, default=5,
                help="Number of rows to compute image grids, both in image query and dataset samples")
    parser.add_argument("-se", "--serialize", action="store_true",
                help="Serialize data to speed-up retrieval.")
    parser.add_argument("-cm", "--comparemethod", default="hellinger", choices=["corr", "chisq", "intersect", "hellinger"],
                help="Histogram compare method. Used only for smart histogram and histogram search.")

    # Parse arguments.
    args = parser.parse_args()
    
    if args.search == "hist":
        img_search = HistogramSearch(args.datapath, serial_hist_path=get_serial_path(args.search, args.serialize))
        results = img_search.search(args.image, args.comparemethod, topk=args.topk)
    elif args.search == "smarthist":
        img_search = HistogramSearch(args.datapath,
                                     serial_hist_path=get_serial_path(
                                                                      args.search, 
                                                                      args.serialize, 
                                                                      hgrid=args.hgrid,
                                                                      wgrid=args.wgrid
                                                                    )
                                    )
        results = img_search.smart_search(args.image, args.comparemethod, topk=args.topk, hgrid=args.hgrid, wgrid=args.wgrid)
    else:
        img_search = ColourSearch(args.datapath, serial_colour_path=get_serial_path(args.search, args.serialize))
        results = img_search.search(load_input_colour(args.colour), topk=args.topk)
    
    plot_img_grid(results)

# directory setting
# cd "Desktop\\Máster\\Information Retrieval, Extraction And Integration\\4. Non textual data extraction\\CBIR"

# Execution examples
# Colour search
# python CBIR.py "pokemon_dataset\\" col -c custom -t 15 -se

# Histogram search
# python CBIR.py "pokemon_dataset\\" hist -i "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg" -t 15 -se -cm hellinger

# Smart histogram search
# python CBIR.py "pokemon_dataset\\" smarthist -i "pokemon_dataset\\Aerodactyl\\d1d381e5f2df42a0973e0251751e1a14.jpg" -t 15 -hg 5 -wg 5 -se -cm hellinger
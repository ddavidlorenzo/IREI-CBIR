import argparse
import colour_search
import hist_search
import numpy as np

# Starting point for CBIR.
if __name__ == '__main__':
    # Argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", help="Path of the folder containing the input files "
                + "to be fed into the CBIR.")
    parser.add_argument("appname", help="Name of the PySpark application to be built.")
    parser.add_argument("-s", "--search", default="dtr", choices=["hist", "col"],
                help='Type of search to conduct. Can be one of: "hist" '
                    + 'for histogram search, "smarthist" for smart histogram search, '
                    + 'or "col" for colour search')
    parser.add_argument("-c", "--colour",
                help='Input colour to CBIR for colour search.')
    parser.add_argument("-i", "--image",
                help="Path to input image to CBIR for histogram search")
    parser.add_argument("-sp", "--serialpath", default=None,
                help="Path to load/store serialized data off/to disk")

    # Parse arguments.
    args = parser.parse_args()

    PREDEFINED_COLOURS = dict(
        yellow = np.array([232, 118, 217]),
        purple = np.array([168, 158, 88]),
        orange = np.array([168, 167, 199]),
        pink = np.array([226, 141, 132]),
        blue = np.array([191, 118, 101])
    )

	# # Check if it is a tree-based regressor.
	# is_tree_based_regressor = args.regressor in ('dtr', 'rfr') 

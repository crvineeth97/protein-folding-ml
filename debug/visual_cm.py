import sys
import numpy as np
from os import listdir
from time import sleep

sys.path.append("../")
from visualize import ContactMap

visualizer = ContactMap()

folder_name = "../data/preprocessed/testing_no_missing/"
input_files = listdir(folder_name)
for filename in input_files:
    mp = np.load(folder_name + filename)["contact_map"]
    visualizer.plot_contact_map(filename, mp)
    sleep(5)

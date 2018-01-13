from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime

def get_data(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    # the first line is the header
    lines = lines[1:]

    x = []
    for line in lines:
        data = np.double(line.split(',')[1:6]);
        x.append(data)  # [1] is the opening price

    return np.array(x)
	



	
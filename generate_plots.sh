#!/usr/bin/env python

## TESTING
import numpy as np
import matplotlib.pyplot as plt
import os
from past.builtins import execfile

os.chdir("_site/plots/")
execfile("burgers_filtered.py")

with open("_site/index.html", 'w') as file:
	file.write("Done generating plots.")
print("Done generating plot scripts.")

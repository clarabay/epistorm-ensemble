import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.interpolate import interp1d
from datetime import timedelta
from epiweeks import Week
import epiweeks
from sodapy import Socrata
from datetime import datetime
from datetime import date, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import seaborn as sns
from scorepi import *
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from ensemble import create_ensemble_method1, create_categorical_ensemble_quantile, create_activity_level_ensemble
from pathlib import Path
import sys
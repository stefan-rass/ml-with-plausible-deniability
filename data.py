# values from execution_snapshot
import numpy as np
import re
from io import StringIO


def parse(x):
    """ Parse matlab-like matrix string into numpy array """
    x = re.sub(r'\n\s+', '\n', x)  # prune leading spaces
    x = re.sub(' +', ',', x.strip())  # replace spaces with commas
    return np.loadtxt(StringIO(x), delimiter=",", dtype=float)


# define symbols, otherwise we get linter errors
B = e = p = p2 = p_orig = p_true = rand_mul = x_decoy = x_training = y_decoy = y_training = None

# Parse output from execution_snapshot.txt, and add them to (exported) locals
with open('execution_snapshot_v2.txt', 'r') as f:  # read file
    data = f.read()  # read only once
    # find all <Name> =\n\n<Matrix>\n
    for name, matrix in re.findall(r'(\w+) =\n\n([\d\s.-]+)\n', data):
        locals()[name] = parse(matrix)
    # also find all <Name> = <Value>
    for name, value in re.findall(r'(\w+) = *([\d.-]+)', data):
        locals()[name] = float(value)

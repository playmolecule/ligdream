# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output", required=True, help="Path to output numpy arrays.")
args = vars(parser.parse_args())


smiles_path = args["input"]
np_save_path = args["output"]


smiles = open(smiles_path, "r").read().strip().split()
strings = np.zeros((len(smiles), 62), dtype='uint8')


vocab_list = ["pad", "start", "end",
    "C", "c", "N", "n", "S", "s", "P", "O", "o",
    "B", "F", "I",
    "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6",
    "#", "=", "-", "(", ")"
]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}


for i, sstring in enumerate(tqdm(smiles)):
    sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
    vals = [1] + [vocab_c2i_v1[xchar] for xchar in sstring] + [2]
    strings[i, :len(vals)] = vals

np.save(np_save_path, strings)

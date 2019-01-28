# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

vocab_list = ["pad", "start", "end",
    "C", "c", "N", "n", "S", "s", "P", "O", "o",
    "B", "F", "I",
    "Cl", "[nH]", "Br", # "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6",
    "#", "=", "-", "(", ")"  # Misc
]

vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

def decode_smiles(in_tensor):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)
    return gen_smiles
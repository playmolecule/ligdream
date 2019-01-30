# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from htmd.molecule.util import uniformRandomRotation
from htmd.smallmol.smallmol import SmallMol
from htmd.molecule.voxeldescriptors import _getOccupancyC, _getGridCenters

import numpy as np
import multiprocessing
import math
import random

vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", # "X", "Y", "Z",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")"  # Misc
]

vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}

vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

resolution = 1.
size = 24
N = [size, size, size]
bbm = (np.zeros(3) - float(size * 1. / 2))
global_centers = _getGridCenters(bbm, N, resolution)


def string_gen_V1(in_string):
    out = in_string.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
    return out


def tokenize_v1(in_string, return_torch=True):
    caption = []
    caption.append(0)
    caption.extend([vocab_c2i_v1[x] for x in in_string])
    caption.append(1)
    if return_torch:
        return torch.Tensor(caption)
    return caption


def get_aromatic_groups(in_mol):
    """
    Obtain groups of aromatic rings
    """
    groups = []
    ring_atoms = in_mol.GetRingInfo().AtomRings()
    for ring_group in ring_atoms:
        if all([in_mol.GetAtomWithIdx(x).GetIsAromatic() for x in ring_group]):
            groups.append(ring_group)
    return groups


def generate_representation(in_smile):
    """
    Makes embeddings of Molecule.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        mol = SmallMol(m)
        return mol
    except:  # Rarely the conformer generation fails
        return None


def generate_sigmas(mol):
    """
    Calculates sigmas for elements as well as pharmacophores.
    Returns sigmas, coordinates and center of ligand.
    """
    coords = mol.getCoords()
    n_atoms = len(coords)
    lig_center = mol.getCenter()

    # Calculate all the channels
    multisigmas = mol._getChannelRadii()[:, [0, 1, 2, 3, 7]]

    aromatic_groups = get_aromatic_groups(mol._mol)
    aromatics = [coords[np.array(a_group)].mean(axis=0) for a_group in aromatic_groups]
    aromatics = np.array(aromatics)
    if len(aromatics) == 0:  # Make sure the shape is correct
        aromatics = aromatics.reshape(aromatics.shape[0], 3)

    # Generate the pharmacophores
    aromatic_loc = aromatics + (np.random.rand(*aromatics.shape) - 0.5)

    acceptor_ph = (multisigmas[:, 2] > 0.01)
    donor_ph = (multisigmas[:, 3] > 0.01)

    # Generate locations
    acc_loc = coords[acceptor_ph]
    acc_loc = acc_loc + (np.random.rand(*acc_loc.shape) - 0.5)
    donor_loc = coords[donor_ph]

    donor_loc = donor_loc + (np.random.rand(*donor_loc.shape) - 0.5)
    coords = np.vstack([coords, aromatic_loc, acc_loc, donor_loc])

    final_sigmas = np.zeros((coords.shape[0], 8))
    final_sigmas[:n_atoms, :5] = multisigmas
    pos1 = n_atoms + len(aromatic_loc)  # aromatics end

    final_sigmas[n_atoms:(pos1), 5] = 2.
    pos2 = pos1 + len(acc_loc)
    final_sigmas[pos1:pos2, 6] = 2.
    final_sigmas[pos2:, 7] = 2.

    return final_sigmas, coords, lig_center


def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """

    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center


def voxelize(multisigmas, coords, center, displacement=2., rotation=True):
    """
    Generates molecule representation.
    """

    # Do the rotation
    if rotation:
        rrot = uniformRandomRotation()  # Rotation
        coords = rotate(coords, rrot, center=center)

    # Do the translation
    center = center + (np.random.rand(3) - 0.5) * 2 * displacement

    centers2D = global_centers + center
    occupancy = _getOccupancyC(coords.astype(np.float32),
                               centers2D.reshape(-1, 3),
                               multisigmas).reshape(size, size, size, 8)
    return occupancy.astype(np.float32).transpose(3, 0, 1, 2,)


def generate_representation_v1(smile):
    """
    Generate voxelized and string representation of a molecule
    """
    # Convert smile to 3D structure

    smile_str = list(smile)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])

    mol = generate_representation(smile_str)
    if mol is None:
        return None

    # Generate sigmas
    sigmas, coords, lig_center = generate_sigmas(mol)

    vox = voxelize(sigmas, coords, lig_center)

    return torch.Tensor(vox), torch.Tensor(smile), end_token + 1


def gather_fn(in_data):
    """
    Collects and creates a batch.
    """
    # Sort a data list by smiles length (descending order)
    in_data.sort(key=lambda x: x[2], reverse=True)
    images, smiles, lengths = zip(*in_data)

    images = torch.stack(images, 0)  # Stack images

    # Merge smiles (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(smile) for smile in smiles]
    targets = torch.zeros(len(smiles), max(lengths)).long()
    for i, smile in enumerate(smiles):
        end = lengths[i]
        targets[i, :end] = smile[:end]
    return images, targets, lengths


class Batch_prep:
    def __init__(self, n_proc=6, mp_pool=None):
        if mp_pool:
            self.mp = mp_pool
        elif n_proc > 1:
            self.mp = multiprocessing.Pool(n_proc)
        else:
            raise NotImplementedError("Use multiprocessing for now!")

    def transform_data(self, smiles):
        inputs = self.mp.map(generate_representation_v1, smiles)

        # Sometimes representation generation fails
        inputs = list(filter(lambda x: x is not None, inputs))
        return gather_fn(inputs)


def queue_datagen(smiles, batch_size=128, n_proc=12, mp_pool=None):
    """
    Continuously produce representations.
    """
    n_batches = math.ceil(len(smiles) / batch_size)
    sh_indencies = np.arange(len(smiles))

    my_batch_prep = Batch_prep(n_proc=n_proc, mp_pool=mp_pool)

    while True:
        np.random.shuffle(sh_indencies)
        for i in range(n_batches):
            batch_idx = sh_indencies[i * batch_size:(i + 1) * batch_size]
            yield my_batch_prep.transform_data(smiles[batch_idx])

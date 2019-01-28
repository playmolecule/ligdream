# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

from .networks import EncoderCNN, DecoderRNN, LigandVAE
from .generators import generate_representation, generate_sigmas, voxelize
from .decoding import decode_smiles
from rdkit import Chem
from torch.autograd import Variable
import torch


def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids


def get_mol_voxels(smile_str):
    """
    Generate voxelized representation of a molecule.
    :param smile_str: string - molecule represented as a string
    :return: list of torch.Tensors
    """
    # Convert smile to 3D structure
    mol = generate_representation(smile_str)
    if mol is None:
        return None

    # Generate sigmas
    sigmas, coords, lig_center = generate_sigmas(mol)
    vox = torch.Tensor(voxelize(sigmas, coords, lig_center))
    return vox[:5], vox[5:]


class CompoundGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False
        self.encoder = EncoderCNN(5)
        self.decoder = DecoderRNN(512, 1024, 29, 1)
        self.vae_model = LigandVAE(use_cuda=use_cuda)

        self.vae_model.eval()
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.vae_model.cuda()
            self.use_cuda = True

    def load_weight(self, vae_weights, encoder_weights, decoder_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.vae_model.load_state_dict(torch.load(vae_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))


    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions = torch.stack(captions, 1)
        if self.use_cuda:
            captions = captions.cpu().data.numpy()
        else:
            captions = captions.data.numpy()
        return decode_smiles(captions)

    def generate_molecules(self, smile_str, n_attemps=300, lam_fact=1., probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param lam_fact: float - latent space pertrubation factor
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """

        shape_input, cond_input = get_mol_voxels(smile_str)
        if self.use_cuda:
            shape_input = shape_input.cuda()
            cond_input = cond_input.cuda()

        shape_input = shape_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1, 1)
        cond_input = cond_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1, 1)

        shape_input = Variable(shape_input, volatile=True)
        cond_input = Variable(cond_input, volatile=True)

        recoded_shapes, _, _ = self.vae_model(shape_input, cond_input, lam_fact)
        smiles = self.caption_shape(recoded_shapes, probab=probab)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)
        return [Chem.MolFromSmiles(x) for x in smiles]

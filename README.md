# LigDream: Shape-Based Compound Generation



## Training 

Note that training runs on a GPU and it will take several days to complete. 

First construct a set of training molecules: 
```
$ python prepare_data.py -i "./path/to/my/smiles.smi" -o "./path/to/my/smiles.npy"  
```
Example training SMILES can be downloaded [here](http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip).

Secondly, execute the training of a model:

```
$ python train.py -i "./path/to/my/smiles.npy" -o "./path/to/models"  
```

## Generation


Web based compund generation is available at [https://playmolecule.org/LigDream/](https://playmolecule.org/LigDream/).

For an example of local novel compound generation please follow notebook `generate.ipynb`.

Trained model weights can be found at this [link](http://pub.htmd.org/ligdream-20190128T143457Z-001.zip).


### Requirements


Model training is written in `pytorch==0.3.1` and uses `keras==2.2.2` for data loaders. `RDKit==2017.09.2.0` and `HTMD==1.13.9` are needed for molecule manipulation.


### License

Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.

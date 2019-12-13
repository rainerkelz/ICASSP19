# ICASSP19

This is code to reproduce the results in [this paper](https://ieeexplore.ieee.org/document/8683582).

## Installation

- it is recommended you first create a python 3 virtual environment

```
$ python3 -m venv ICASSP19
$ cd ICASSP19
$ source bin/activate
$ git clone https://github.com/rainerkelz/ICASSP19
```

- until a new [madmom](https://github.com/CPJKU/madmom) version is released on pip, you'll have to build [madmom](https://github.com/CPJKU/madmom) from source:

```
$ pip install -r ICASSP19/requirements_00.txt
$ git clone https://github.com/CPJKU/madmom.git
$ cd madmom
$ git submodule update --init --remote
$ python setup.py develop
$ cd ..
```

- you should have [madmom](https://github.com/CPJKU/madmom) version 0.17.dev0 or higher now
- now we'll install the second set of requirements

```
$ pip install -r ICASSP19/requirements_01.txt
```

## Data
- obtain the [MAPS](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) dataset

- create datadirectory, and symlink to MAPS data
```
$ mkdir data
$ cd data
$ ln -s <path-to-where-MAPS-was-extracted-to> .
$ cd ..
```

- create metadata-file for non-overlapping MAPS MUS subset (or use the one checked in ...)
```
$ python create-maps-metadata-file-non-overlapping.py data/maps_piano/data
```

## Training
- start training with
```
$ python run.py configs/ofos.py --train --cuda
```
- the learnrate is maybe set a tiny little bit too high, and you may need to restart the script a few times until it actually learns properly (thanks to Adrien Ycart for reminding me!)
- as an alternative, you could use a learnrate scheduler that starts with a lower learnrate and speeds up after a few batches / epochs (we have **not** tested this!)

## Gridsearch, Finetuning
- some editing required here ...
- export all activations for the training pieces
```
$ mkdir -p exported/ofos/lr_0.15_bs_256/train
$ python export.py runs/ofos/lr_0.15_bs_256/best_valid_loss.pkl train exported/ofos/lr_0.15_bs_256/train --cuda
```

- run gridsearch
```
$ python gridsearch.py exported/ofos/lr_0.15_bs_256/train exported/ofos/lr_0.15_bs_256/results_train.pkl
```

- after we find nice values for the few parameters we have, we replace the gridsearch with these, and export the activations on the test pieces, then run the HMM on the test pieces to obtain our results
```
$ mkdir -p exported/ofos/lr_0.15_bs_256/test
$ python export.py runs/ofos/lr_0.15_bs_256/best_valid_loss.pkl test exported/ofos/lr_0.15_bs_256/test --cuda
```

- run test (WITH THE **ONE** SET OF PARAMETERS, OBTAINED IN THE PREVIOUS STEP!)
```
$ python gridsearch.py exported/ofos/lr_0.15_bs_256/test exported/ofos/lr_0.15_bs_256/result_test.pkl
```

## Export your newly trained model to madmom
- in case you want your pytorch model converted to [madmom](https://github.com/CPJKU/madmom), so it can be deployed with less dependencies, you can use the `convert.py` script
```
$ python convert.py runs/ofos/lr_0.15_bs_256/best_valid_loss.pkl notes_cnn.pkl
```
- if you give the converter script an audiofilename, it runs the whole pre-processing chain (which needs to be adapted to what was in the config used for training) for the audiofile, and then applies the converted model, and finally displays the output feature maps

- by now, there should be a trained model available in [madmom](https://github.com/CPJKU/madmom) that achieves around ~57.XX f-measure on the MAPS test set as described in the paper

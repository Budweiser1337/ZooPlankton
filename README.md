## Overview
The objective of this project is to segment living organisms from non living stuff. The stuff can be non living body parts, debris etc...

The segmentation is to be performed on scanned images from the ZooPlankton device. The images have been collected at Villefranche-sur-mer and provided by

Sorbonne Université/CNRS - Institut de la Mer de Villefranche (IMEV), Sorbonne Université/CNRS - Laboratoire d'Océanographie de Villefranche (LOV); 2020; Plankton community in Régent (680µm) net, Point B, Villefranche-sur-Mer, France https://dx.doi.org/10.14284/477

The data have been collected with a Régent (680µm) net and contains organisms whose size ranges from ~700µm up to ~5cm. These organisms belong to mesozooplankton. With these observations, the researchers seek to better understand the dynamics of these communities.

The scanners has a high throughput rendering difficult the labeling by humans. Hence, the researchers are interested in automating the process of labeling the scans. The tasks we will focus on in this challenge is the task of segmenting living from non-living, with the objective to isolate living organisms for possibly later classification of the segmented individuals.

This project was made to compete in a private kaggle competition, which was part of a deep learning class at CentraleSupélec.

## Usage

### Local experimentation

For a local experimentation, you start by setting up the environment :

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

Then you can run a training, by editing the yaml file, then 

```
python -m torchtmpl/main.py config.yml train
```

And for testing (**not yet implemented**)

```
python torchtmpl/main.py config.yaml test
```

### Cluster experimentation

For running the code on a cluster, we provide an example script for starting an experimentation on a SLURM based cluster.

The script we provide is dedicated to a use on our clusters and you may need to adapt it to your setting. 

Then running the simulation can be as simple as :

```
python3 submit-slurm.py config.yaml
```

## Testing the functions

Every module/script is equiped with some test functions. Although these are not unitary tests per se, they nonetheless illustrate how to test the provided functions.

For example, you can call :


```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
python -m torchtmpl.models
```

and this will call the test functions in the `torchtmpl/models/__main__.py` script.


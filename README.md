# RPNODE_FSS
Robust Prototypical Few-Shot Organ Segmentation with Regularized Neural-ODEs


<p align="center">
  <img src="regularisation.jpg" width="80%"/><br>
</p>

## Installation and setup

To install this repository and its dependent packages, run the following.

```
git clone https://github.com/rpnode-fss/RPNODE_FSS.git
cd RPNODE_FSS
conda create --name RPNODE_FSS # (optional, for making a conda environment)
pip install -r requirements.txt
```

The processed datasets can be downloaded from [here](https://drive.google.com/drive/folders/1o2yiBOKzkwxsSc-gWWwE_Z0yJiWtenMa?usp=sharing).

Some relevant trained model weights can be downloaded from [here](https://drive.google.com/drive/folders/1oUdQ-mDndbCiiWQSdMX_ihhzPYWMGK8A?usp=sharing).

Change the paths to BCV, CT-ORG and Decathlon datasets in  `config.py` and  `test_config.py` according to paths on your local. Also change the path to ImageNet pretrained VGG model weights in these files.



## Training

To train  R-PNODE,  run

```
python3 train.py with model_name=<save-name> target=<test-target> n_shot=<shot> ode_layers=3 ode_time=4
```

Further parameters like the standard deviation of gaussian perturbation can be changed in the configs. 

## Testing

To test a trained model, run

```
python3 test_attacked.py with snapshot=<weights-path> target=<test-target> dataset=<BCV/CTORG/Decathlon> attack=<Clean/FGSM/PGD/SMIA> attack_eps=<eps> to_attack=<q/s>
```

This command can be used for testing on all settings, namely 1-shot and 3-shot, liver  and  spleen and Clean, FGSM, PGD and SMIA with different epsilons. 


### Class Mapping

```
BCV:
    Liver: 6
    Spleen: 1
CT-ORG: 
    Liver: 1
Decathlon: 
    Liver: 2
    Spleen: 6
```

# NN4Fluid
Code Repo for using neural networks for Fluid Simulation and Simulation generation scripts in mantaflo

## General Instruction for the learner

All right, you have now gained access the repo. Before you start executing the code, the make sure you understant the flow of data within the code. 

Start reading from main.py then continue to wherever the functions lead you(Mostly it will lead you to trainer.py :D). Once you understand the code flow it would be easier for you to make changes and build further onto it.

## Installations

### Python Requirements

All the python packages required are mentioned in requirements.txt and can be installed using 

``` pip install -r requirments.txt```

Make sure you system supports GPU and had CUDA, and tensorflow-gpu is installed correctly.


### Mantaflow

To install `mantaflow`, run:

    $ git clone https://bitbucket.org/mantaflow/manta.git

The current head of mantaflow has bugs with numpy support so checkout 15eaf4 before following the steps mentioned in [instruction](http://mantaflow.com/install.html). 
    $ git checkout 15eaf4
    
Also while installing keep the `-DNUMPY='ON'` option enabled

## Usage

### Data Generation

To start generating data execute `liquid3_x_y.py` for Dam break or `smoke_pos_size.py` for smoke plume using mantaflow.
    $ <mantaflow_directory>/build/manta smoke_pos_size.py

### Training 

To traing your model you can use `train.sh` or the following command

    $ python main.py --is_3d=False --dataset=liquid3_x10_y10_f200 --res_x=256 --res_y=128 --gpu_id=1

Make sure the dataset argument has the correct name for the folder in which data is stored.

This would put all the relevant files and folder in `log/<dataset>/` folder which include 
- Intermediate  results in .png format
- Saved weights in .ckpt format
- Tensorboard files starting with events.out....

### Using The Neural Networks

To genrate the respective veloctiy field in .npz files run `test.sh` or the following command

    $ python main.py --is_train=False --load_path=log/liquid3_x10_y10_f200/0623_225552_de_tag/ --use_curl=False --is_3d=False --dataset=liquid3_x10_y10_f200 --res_x=256 --res_y=128 --gpu_id=1 

This will put the files in a folder created in `log/liquid3_x10_y10_f200/' 

Copy this files into postTrainging folder. Then execute

    $ python liquid_3_x_y.py
in the postTraining folder(this file is different from the file in scene folder).

This would generated .png files in 'l_adv/' folder, which you can use to analyze the results or create videos using frms2vid.py from Localfiles

##### Pointers:

- Make sure the values of p1,p2 in trainer.py 's test function are noted down because you would need to keep these same parameters in reconstructing the simulation using `postTrainging/liquid_3_x_y.py`
-  use the same values p1 and p2 in liquid_3_x_y.py 's advect function.

## Index (Folder Structure)

the data files are stored in /wd/users/b16029/MTP_LargeFolders

### /log
All the filed generated while training and testing are saved in here. This includes:
- Intermediate  results in .png format
- Saved weights in .ckpt format
- Tensorboard files starting with events.out....

### /data
All the simulation data generated is saved here with respective dataset name

### /scene

All the simulation scripts are stored in /scene folder

### /localfiles

All the files needed to create results are stored in localFiles, which can be run on one's local system.

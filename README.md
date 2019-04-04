# Question-Answering Networks

This repo implements a set of question-answering networks 
on the [SQuAD dataset] (https://rajpurkar.github.io/SQuAD-explorer/). 
They range from baselines, using just a 
dense layer or an lstm as a decoder up to models which were
state-of-the-art at the time of publication. The models are:
  - Base-Model with Dense Layer as Decoder
  - LSTM-Model with a BiLSTM Layer plus an implementation of Pointer Net as Decoder
  - [MatchLSTM] (https://arxiv.org/abs/1608.07905)
  - [RNet] (https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
  - [QANet] (https://arxiv.org/abs/1804.09541) 
 
While they since have been improved on using transfer-learning 
approaches such as [ELMo] (https://github.com/allenai/bilm-tf), 
[OpenAI GPT] (https://github.com/openai/finetune-transformer-lm)
or [Bert] (https://github.com/google-research/bert) I hope they
still provide a useful reference for interested practitioners.

We used preprocessing and layer implementations from a set of other great 
repos implementing those models which are:
  - https://github.com/HKUST-KnowComp/R-Net
  - https://github.com/NLPLearn
  - https://github.com/MurtyShikhar/Question-Answering
  
## Requirements
This repo has been tested using:
  - numpy==1.16.2
  - tabulate==0.8.3
  - tqdm==4.31.1
  - bottle==0.12.16
  - spacy==2.0.18
  - pyyaml==5.1
  - tensorflow-gpu==1.12
  
## Download and Setup
First, you need to download the SQuAD dataset and the Glove word embeddings and
store them in folders ```datasets/squad``` and ```datasets/glove``` respectively.
You can do this by running
```shell
$ ./download.sh
```
in your terminal.

Next, you need to install the necessary python packages. I have provided
a Pipfile for this purpose. To be able to use it, you need to have
pipenv installed on your system. To do this, run
```shell
$ pip install -U pipenv
```
The Pipfile includes the path to the tensorflow wheel with gpu support.
This is recommended to train the models. If you want to build tensorflow
without gpu-support you can comment out the relevant line and un-comment
the line below, which is the path to the tensorflow wheel without gpu-support.

Once this is done, you can create a virtual environment with all necessary
packages by executing
```shell
$ pipenv install
```
You can access the virtualenv using
```shell
$ pipenv shell
```
Once you're done, exit it by typing
```shell
$ exit
```
If you have cuda installed on your system, you can run the models within
this virtual environment.

You can also try running the models in a docker container, which can be 
created using the Dockerfile in this repo. Build the Docker Image using
```shell
$ docker build -t nlp-tf1.12 .
```
if you have Nvidia-Docker installed (follow the instructions [here] 
(https://docs.docker.com/install/) and [here] (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
 pto install it). Once this is done start
the docker container using
```shell
$ docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v /path/to/repo/question-answering:/question-answering -it nlp-tf1.12 /bin/bash 
``` 
Change to the directory of the repo,
```shell
$ cd /question-answering
```
and start training/predicting!

## Training
Each model includes a config.yaml file with all parameters. Parameters which have the 
value "-1" are not relevant for this particular model. Any paths or parameter values 
can be changed using this files. 
To train any of the models, run the following command in your docker container:
```shell
$ python3 train.py --config [model_name]/config.yaml
```
This will start training and print the current state on screen as well
as into an output file called "phrase_level_qa". If the current epoch 
improves the score, a checkpoint of the model is saved into out_dir defined in config.yaml.

## Testing
To get the results on the test set you have to specify the folder
in the corresponding config file which the model checkpoint you want to test 
is stored in. The parameter in the config.yml file is called ```use_out_dir```. 
For example if the model checkpoint is stored in folder ```20190403083913```
specify this in the config file.

Parameter ```checkpoint``` specifies which checkpoint should be loaded. If it is 
an empty string the last checkpoint will be loaded. If you want to load
checkpoint ```model-9``` you would specify '9' in config.yml.

Once this is done run
```shell
$ python3 test.py --config [model_name]/config.yaml
```
This will display the result on screen and in the log file. It will 
also save the predictions in a json file called ```answer.json```. This
file can be used to compute the score using the official SQuAD evaluation
script. This can be done by running
```shell
$ python3 evaluate-v1.1.py ./data/squad/dev-v1.1.json ./runs/[model_name]/answer.json
```

## Results
Here I've collected some results from the various implementations. All experiments
were run on a NVIDIA 

|      Model     | Training Epochs | Size |  EM   |  F1   | train-time (hrs) |
|:--------------:|:---------------:|:----:|:-----:|:-----:|:----------------:|
|      base-model|       30        |  150 | 26.65 | 37.27 |      ~18         |
|            lstm|       30        |  150 | 43.68 | 54.12 |      ~30         |
|      match-lstm|       30        |  150 | 60.31 | 70.26 |      ~35         |
|cudnn-match-lstm|       50        |  150 | 60.43 | 70.32 |      ~12         |
|          QA-Net|       50        |  128 | 67.81 | 77.74 |      ~15         |
|           R-Net|       30        |  150 |       |       |      ~38         |
|     cudnn-R-Net|       50        |  32 | 68.84 | 78.14 |       ~8         |
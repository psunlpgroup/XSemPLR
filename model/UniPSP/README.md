# Multilingual Pretrained Encoder-Decoder Models (Enc-Dec)
This folder keeps the repo for a group of Enc-Dec models, mainly including mBERT+PTR and XLM-R+PTR
## Model Overview
The second group uses pretrained encoder-decoder models, including mBART (Chip-
man et al., 2022) and mT5 (Xue et al., 2020) which uses text-to-text denoising objective for pretraining over multilingual corpora.
## Dependency

To establish the environment run this code in the shell (the third line is for CUDA11.1):

``````
conda env create -f py3.7pytorch1.8.yaml
conda activate py3.7pytorch1.8
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
``````

That will create the environment `py3.7pytorch1.8` we used. 

Then, for now, you need to execute another command in shell to download the some nltk data into the env
 if you haven't download it which we will use in experiments.

``````
python -m nltk.downloader -d YOUR/CONDA/ENV/share/nltk_data punkt stopwords
``````

## Run the Model
1. Prepare the dataset. For monolingual training, one can directly use the `dataset` folder in root folder. If you need to run other settings, such as few-shot, or multilingual settings, one needs to `python /utils/few-shot.py` (utils in root folder) to create a few-shot version of dataset.
2. Adjust running script. Please check line 32 of `tasks/{DATASETNAME_SETTING}.py` to ensure the path is correct.
3. Run the script. `bash scripts/{MODEL}/{SETTING}/run_{DATASETNAME}_{LANGUAGE}.sh`. e.g., `bash scripts/mT5/monolingual/run_matis_de.sh`. One may need to modify the script before usage, such as GPU options.
## Introduction of each file

### [configure](https://github.com/Timothyxxx/UniPSP/tree/master/configure)
Code for configuration of different tasks/settings,
more details see README in [./configure](https://github.com/Timothyxxx/UniPSP/tree/master/configure)
 
### [metrics](https://github.com/Timothyxxx/UniPSP/tree/master/metrics)
Code for evaluating the prediction of our model,
more details see README in [./metrics](https://github.com/Timothyxxx/UniPSP/tree/master/metrics)

### [models](https://github.com/Timothyxxx/UniPSP/tree/master/models)
Code for models(for now, we have seq2seq models(mT5 and mBART).

### [seq2seq_construction](https://github.com/Timothyxxx/UniPSP/tree/master/seq2seq_construction)
Code for evaluating the prediction of our model,
more details see README in [./seq2seq_construction](https://github.com/Timothyxxx/UniPSP/tree/master/seq2seq_construction)

### [third_party](https://github.com/Timothyxxx/UniPSP/tree/master/third_party)
packages from the third party for us to tmp store, and we will redirect them by git recursive deployment in the end. 

### [utils](https://github.com/Timothyxxx/UniPSP/tree/master/utils)
Code for some useful(or not) stuff, it contains:
- **configure.py**: The "args" data-structure for **parsing and store the config files** in ./configure. (and we are trying to change it 
into a more main-stream structure which also support read from the file and create nested config object.)
- **dataset.py**: Wrap the seq2seq dataset to tokenize the "seq_in" and "seq_out", since the trainer only support tokenized tensors of "seq_input" and "seq_output" as input
- **tool.py**: The tool to **get datasets, models and evaluators in a unified way**. 
- **trainer.py**: The **modified trainer version** of huggingface trainer class **for supporting meta-tuning**(we want get our training sampler under our control), 
**easier evaluation**(the metrics of huggingface's input format(numbers) is contradicted with ones of all official evaluations)
 and **further changes in this project**(for example, we want to feed more para in a model.forward function).
- **training_arguments.py**: The **customized wrapped training_arguments**.

### train.py
- together with the config file, act as the start and main-control of the experiment.

### Procedure
The working procedure of our work follows:
raw data(s) -> + seq2seq data(s) ("seq_in" and "seq_out") -> tokenized -> seq2seq_trainer -> predictions -> eval(s)

## The overview file structure of this Unified Framework
```
.
├── configure                          # Code for configuration of different tasks/settings
│   └── XSP                            # config files for running XSemPLR, containing vairous cfg file for mBART and mT5 models.
├── metrics                            # Code for evaluating the prediction of our model
├── models                             # Code for models
│   └── unified
│       └── finetune.py                # model of the bare finetune
├── seq2seq_construction               # Code for wrap the raw data into seq_in and seq_out and add them
│   ├── ...                            # check the README in the ./seq2seq_construction
├── tasks                              # Code for encoder-decoder architecture
│   ├── ...                            # check the README in the ./tasks
├── third_party                        # packages from the third party
│   ├── ...                            # if you use any github repo from others, try to put them in this dir, and note the links in the .submodules for us to make them easier to e added by the recursive clone of git.
├── utils                              # Code for some useful(or not) stuff
│   ├── __init__.py             
│   ├── configure.py                   # the util for parsing the cfg file in ./configure, will get nested args object which is human friendly.
│   ├── dataset.py                     # wrap the seq2seq dataset constructed, tokenize the seq_in and seq_out for feed into the trainer.
│   ├── tool.py                        # Use the reflection to help loading model, seq2seq constructor and evaluator
│   ├── trainer.py                     # we changed the original trainer into the EvaluationFriendlyTrainer in this file, for easier eval, also we controled the sequence of trainer to be in original order, and add description in it, if you want make any modifications in forward of the models, you may need to change something in here.
│   └── training_arguments.py          # wrapped training arguments for seq2seq
├── .gitignore                         # use to ignored some tast or debug files in your desktop
├── .gitmodules                        # use the recursive clone of the git, will be used to create files in ./third_party at last
├── auto_shell.py                      # help you run the model with schedule on GPU
├── README.md                          # As you can see, this is the README 
└── train.py                           # The start of the code, control the train, eval, test, store and log and 
```


## Add a new dataset into the framework

For each dataset, please follow these steps:

- Step 1, Add the loader of raw data in `./tasks`. Check `./tasks/mgeoquery.py` for an example.

- Step 2, Add the wrapper in `./seq2seq_construction`. This constructs "seq_in" and "seq_out". Check `./seq2seq_construction/mgeoquery.py` for an example.

- Step 3, Add the config file in `./configure/XSP/`. It defines some configuration for your experiments. Check `./configure/XSP/mt5_mgeoquery.cfg` for an example. You can focus on t5-large for English first; then we will try mBART and mT5 on other languages.

- Step 4, Add a shell script for running the experiment. Check `./scripts/mT5/monolingual/run_mgeoquery.sh` for an example.

### With [wandb](https://wandb.ai/) report
[wandb platform](https://docs.wandb.ai/) is a useful tool for us to track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings. We highly recommand using this platform to versialize, track and share results. 

To use that in our framework, you can change the env parameter WANDB_API_KEY, WANDB_PROJECT(you need to have a quick register on the wandb platform).
``````shell
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
``````
Check `./scripts/mT5/monolingual/run_mgeoquery.sh` for an example.

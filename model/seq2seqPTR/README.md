# Multilingual Pretrained Encoders with Pointer-based Decoders (Enc-PTR)
This folder keeps the repo for a group of Enc-PTR models, mainly including mBERT+PTR and XLM-R+PTR

## Model Overview
This group of models is
multilingual pretrained encoders with decoders aug-
mented with pointers. Both encoders and decoders
use Transformers (Vaswani et al., 2017). The de-
coder uses pointers to copy entities from natural
language inputs to generate meaning representa-
tions (Rongali et al., 2020; Prakash et al., 2020).
We use two types of multilingual pretrained en-
coders, mBERT (Devlin et al., 2018) and XLM-
R (Conneau et al., 2019), and both are trained on
web data covering over 100 languages

## Environment Setup
```
conda create -n xsp python=3.7
conda activate xsp

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install seqeval
pip install num2word
pip install us
pip install airportsdata
pip install wandb
pip install sentencepiece
```

## Run the Model
1. Adjust the script to run the model, such as the path of the dataset, and `CUDA_VISIBLE_DEVICES`. The scripts are in `\scripts` (one can create a new script following any other ones).
2. Run the script, for instance, `./scripts/xlm/few-shot/run_mtop.sh`

## Project Structure
This figure shows the overview of project sturcture
```
.
├── eval_scripts       // Additional scripts to evaluate the predicted results. Main evaluation code is evaluate.py under root                     
├── scripts            // Shells to run the models.                              
├── src                // Models of Enc-PTR, including mBERT+PTR, XLM-R+PTR on 6 settings.
│   ├── data.py        // Load the raw data, and parse them into the input format of this model 
│   ├── ptrbert.py     // Model structure. Implementation of pointers and transformers.  
│   └── utils.py       // Helper functions for model and loading process.                 
├── data_preprocess.py // Precess MSpider dataset and concat tables and queries.             
├── evalute.py         // Evaluate the predicted results.     
├── postprocess_eval.py// Postprocess of spider dataset and evalute the results.
├── preprocess.py      // Prerpcess datasets other than spider.
└── ptrnet_bert.py     // Entry of the project, containing training and main logics.
```
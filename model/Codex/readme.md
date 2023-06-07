## Codex
This folder contains the code for running Codex on XSemPLR.

### Model Overview
Codex is a large model based on GPT. It is fine-tuned on publicly available code from GitHub. While it is not trained on a multilingual corpus, it has shown cross-lingual
semantic parsing capabilities. We use in-context learning with 8 examples (dynamic shirk in case of exceeding input limit.) 

### Example Input Prompt
We list an example from MSpider dataset. Other datasets have similar input but do not have table information.
```
# Translate the following sentences into sql:

# Question:
# Who performed the song named "Le Pop"?

# The information of tables:
# 0. Table name is: Songs. The table columns are as follows: SongId, Title
# 1. Table name is: Albums. The table columns are as follows: AId, Title, Year, Label, Type
# 2. Table name is: Band. The table columns are 
---- 3 Tables Ignored ----
# 6. Table name is: Vocals. The table columns are as follows: SongId, Bandmate, Type

# Translation results are as follows:
# SELECT T2.firstname ,  T2.lastname FROM Performance AS T1 JOIN Band AS T2 ON T1.bandmate  =  T2.id JOIN Songs AS T3 ON T3.SongId  =  T1.SongId WHERE T3.Title  =  "Le Pop"

---- More Examples Ignored ---- 

# Translate the following sentences into sql:

# Question:
# Tell me the types of the policy used by the customer named "Dayana Robel".

# The information of tables:
---- 6 Tables Ignored ---- 

# Translation results are as follows:
```
### Running
- First, set up the environment `pip install openai, retrying`
- Next, modify `\scripts\[LINGUAL]\[DATASET_NAME].sh`, note that you need an OpenAI key for running this. (Codex is deprecated now, thus, model needs to be changed as well.)
- Third, start running, e.g. for monolingual mspider dataset `bash scripts\monolingual\mspider.sh`
- Finally, check the results in `results` folder.



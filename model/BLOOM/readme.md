## BLOOM
This folder contains the code for running BLOOM on XSemPLR.

### Model Overview
BLOOM is a 176B-parameter multilingual language model pretrained on 46 natural and 13 programming languages from the ROOTS corpus. 

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
- First, set up the environment `pip install retrying`
- Next, modify `\scripts\[LINGUAL]\[DATASET_NAME].sh`, note that you need a Huggingface key for running this. (A subscription may increase the quote limit, however may lead to additional cost.)
- Third, start running, e.g. for monolingual MSpider dataset `bash scripts\monolingual\mspider.sh`
- Finally, check the results in `results` folder.



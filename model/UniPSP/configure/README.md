#Configure part
## Why use config?
Since it is a very large project contains **a huge variety of tasks** related to the structure background knowledge,
 the unification of coding is also a challenge faced by all of us.

So we choose to use the configure file to drive specific task/setting. 
We utilise the config file to store the **specific info(which is also relatively fixed)** in the config files.
While the **flexible training arguments** to be stored in the huggingface's training_args.
These two kind of args have their own jobs. 

### UNIFIED, add more tasks to utilize:
It contains the config of specific task/setting which uses different models(prefix-tuning model/finetune model) we have now. 
And you can always **add config file with new settings** after you finished your adding in settings of this project,
Or add some new config parameter into the config file if you want to use them in anywhere of you task.

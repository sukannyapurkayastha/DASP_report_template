# README
We use the same file main.py to train different models:
* T5
* BART
* Pegasus

The data can be created in multiple forms. You need to do the following to generate different splits on the data:
1. Split on attitude roots. This will save the data in ```data``` folder
``` python
python create_data.py
```

2. Start by creating a new conda envirionment:
```python
conda create --name jitsupeer python=3.8
```

And activate it:
```python
conda activate jitsupeer
```

Install requirements:
```python
pip install -r requirements.txt
```
The env works in my setup but I am not so sure how good is the freeze requirements.txt. Just give it a try, if it doesn't work then install libs mentioned in main.py, that's how I did it, since the requirements.txt in original repo doesn't work at all, even version numbers are wrong.

3. results you can find in t5-large-output. To visualise the predictions of test file, you can use vis_test_pred.ipynb
   


4. For finetuning, change the train test and dev files as needed for each of the folders
For BART/PEGASUS ((google/pegasus-large)), we use following commands. We have an example bash file contains training command and other parameters 
```
sbatch run_main.sh
```
Please change folder name when different models are chosen.
```
python main.py \
--model_name_or_path facebook/bart-large \ 
--do_train \
--do_eval \
--do_predict \
--seed 42 \
--num_beams 5 \
--train_file  'train.csv'\
--validation_file 'dev.csv'\
--test_file  'test.csv'\
--output_dir bart/$epochs/$learning_rate \
--overwrite_output_dir \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps 4 \
--num_train_epochs $epochs \
--learning_rate $learning_rate \
--save_steps -1 \
--report_to 'none' \
--predict_with_generate
```

For T5, we use the command,
``` python    
python main.py \
--model_name_or_path t5-large \
--do_train \
--do_eval \
--do_predict \
--num_beams 5 \
--prefix "Generate Description" \
--train_file  'train.csv'\
--validation_file 'dev.csv'\
--test_file  'test.csv'\
--output_dir bart/$epochs/$learning_rate \
--overwrite_output_dir \
--per_device_train_batch_size=8 \
 --per_device_eval_batch_size=8 \
--gradient_accumulation_steps 4 \
--num_train_epochs $epochs \
--learning_rate $learning_rate \
--save_steps -1 \
 --report_to 'none' \
--predict_with_generate
```

5. We change the train and validation file for few-shot setups from the folder ```few_shot```. We use the test set created in Step 2.
* ```1_shot``` (contains 1 shot train,valid files)
* ```2_shot``` (contains 2_shot train,valid files)

We run the same finetuning codes as in Step 2.


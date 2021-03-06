## Simplified Codes for fine-tuning Bert on kinds of tasks
​    Many codes are too complex to understand easily. This project aims to show the procedure of finetuning Bert and solve other task easily! 

## Support Task Type:
1. Sentence-level classification including single sentence classification and sentence pair classification tasks.  
   		E.g. Sentiment Analysis, SNLI and  etc.
2. Token-level classification (sequence labeling task).  
   		E.g. NER, POS and etc.
3. Span extraction task.  
        To do

## Usage:
  Taking the MRPC task as example:
1. Download the dataset from [GLUE](https://gluebenchmark.com/) and add it to the dir: "data/MRPC" including three files named train.tsv/dev.tsv/test.tsv.
2. Download pretrained Bert from [Google-Bert-Uncased-Base](https://github.com/google-research/bert), and unzip it. Run the script "convert_tf2pt.sh" located in dir: "bert"
3. Run the fine-tune codes:  
   (1).Distributed Data Parallel Training:  
      $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 finetune_v2.py --data_dir=data/MRPC/ --task_name=mrpc --do_train --do_predict  
   (2).Data Parallel Training:  
      $ CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_v2.py --data_dir=data/MRPC/ --task_name=mrpc --do_train --do_predict  
   (3).Single GPU Training:  
      $ CUDA_VISIBLE_DEVICES=3 python finetune_v2.py --data_dir=data/MRPC/ --task_name=mrpc --do_train --do_predict  
4. Attention: the first two methods support half precision training which can reduce the GPU memory use. Just install the [Apex](https://github.com/NVIDIA/apex) and add the aurgument "--ft16". The "Distributed Data Parallel Training" method may encounter some errors. If so, pleasure the "Data Parallel Training" method.

## How to use this project for other task?
​    Writing a dataloader for task-specific data. Examples can be found at: "dataset.py"

## To do
1. Finishing the finetune codes for span extraction task and pretrained LM.
2. Summarizing the tricks of finetune.

## Acknowledgement
All codes of this project are based on pytorch-pretrained-bert which is the first version of [Transformers](https://github.com/huggingface/transformers). Thanks for all authors of Transformers. 


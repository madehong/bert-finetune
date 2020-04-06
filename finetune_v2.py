#encoding=utf-8
import argparse
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn import metrics

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam

from dataset import Data

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir", default='',    type=str, required=True, help="The input data dir. Containing three files: train.tsv/dev.tsv/test.tsv files for the task.")
	parser.add_argument("--task_name",default='CoLA',type=str, required=True, help="The task to be fine-tuned.")
	## Other parameters
	parser.add_argument("--max_seq_len",                 default=128,   type=int, help="The maximum total input sequence length after WordPiece tokenization.")
	parser.add_argument("--is_sequence_labeling",        default=False, action='store_true', help="Is sequence labeling task?")
	parser.add_argument("--do_train",                    default=False, action='store_true', help="Whether to run training.")
	parser.add_argument("--do_eval",                     default=False, action='store_true', help="Whether to run eval on the dev set.")
	parser.add_argument("--batch_size",                  default=32,    type=int,   help="Total batch size for training & eval & testing.")
	parser.add_argument("--learning_rate",               default=5e-5,  type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",            default=3.0,   type=float, help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",           default=0.1,   type=float, help="Proportion of training to perform linear LR warmup for. E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",                     default=False, action='store_true', help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",                  default=-1,    type=int, help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',                        default=42,    type=int, help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps', default=1,     type=int, help="Number of updates steps to accumualte before performing a backward/update pass.") 
	parser.add_argument('--optimize_on_cpu',             default=False, action='store_true', help="Whether to perform optimization and keep the optimizer averages on CPU")
	parser.add_argument('--fp16',                        default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--fp16_level',                  default='O1',  type=str,   help="One of ['O0', 'O1', 'O2', 'O3']")
	parser.add_argument('--pretrained_weights_dir',      default='pretrained_weights/',type=str,  help='Dir path to store pretrained model weights including three files: vocab.txt/bert_config.json/pytorch_model.bin')
	## Parse all arguments
	args = parser.parse_args()

	# For distributed training
	if args.local_rank == -1:
		device = torch.device('cuda')
		n_gpu = torch.cuda.device_count()
		if n_gpu > 1:
			logging.info('Data Parallel Training....')
		else:
			logging.info('Single GPU Training....')
	else:
		torch.cuda.set_device(args.local_rank) # This is important
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		torch.distributed.init_process_group(backend='nccl')
		logging.info('Distribution Training....')

	## Fix ths random seed
	seed = args.seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	if n_gpu > 1:
		torch.cuda.manual_seed_all(args.seed)

	## Import proper model
	if args.is_sequence_labeling:
		from model import BertForSequenceLabeling as Model
	else:
		from model import BertForSequenceClassification as Model

	logging.info('Building tokenizer...')
	tokenizer = BertTokenizer(args.pretrained_weights_dir+'vocab.txt')
	
	logging.info('Loading data...')
	data = Data(args.task_name,
				args.data_dir,
				tokenizer,
				args.max_seq_len,
				args.is_sequence_labeling)

	logging.info('Building Model...')
	model = Model.from_pretrained(args.pretrained_weights_dir, data.label_size)
	model.to(device)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
		]

	num_train_steps = None
	if args.do_train:
		num_train_steps = int(len(data.train_data) / args.batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
	
	args.batch_size = int(args.batch_size / args.gradient_accumulation_steps) * n_gpu

	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr=args.learning_rate,
						 warmup=args.warmup_proportion,
						 t_total=num_train_steps)

	## Using half precision for faster training
	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Haven't install apex!!!")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_level)
	
	# For distributed training
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
	if n_gpu > 1:
		model = torch.nn.DataParallel(model)
	
	if args.do_train:
		model.train()
		logging.info('Begin Training...')
		if args.local_rank != -1:
			train_sampler = DistributedSampler(data.train_data)
		else:
			train_sampler = RandomSampler(data.train_data)
		train_loader = DataLoader(data.train_data, sampler=train_sampler, batch_size=int(args.batch_size))#, num_workers=3)
		for epoch in trange(int(args.num_train_epochs), desc='Train Epoch'):
			sum_loss, num_step = 0, 0
			for step, batch in enumerate(tqdm(train_loader, desc='Train Iter')):

				batch = (b.to(device) for b in batch)#inputs, segment, mask, labels = batch[:4]
				loss, _ = model(*batch)

				# Single-Machine Multi-GPU
				if n_gpu > 1:
					loss = loss.mean()

				# Gradient Accumulation
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				# Half Precision 
				if args.fp16:
					with amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()

				sum_loss += loss.item()
				num_step += 1

				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
				model.zero_grad()
				if step  and step % 20 == 0:
					logging.info('Epoch: {}; Step: {}; Avg Loss: {}'.format(epoch, step, sum_loss/num_step))

	if args.do_eval:
		model.eval()
		logging.info('Begining Eval:')
		eval_loader = DataLoader(data.dev_data, batch_size=int(args.batch_size))
		pred, true = [], []
		for step, batch in enumerate(tqdm(eval_loader, desc='Eval Iter')):
			batch = (b.to(device) for b in batch)
			with torch.no_grad():
				if args.is_sequence_labeling:
					inputs_ids, valid_ids, segment_ids, attention_mask, labels = batch
					p, _ = model(inputs_ids, valid_ids, segment_ids, attention_mask)
				else:
					inputs_ids, segment_ids, attention_mask, labels = batch
					p, _ = model(inputs_ids, segment_ids, attention_mask)
			pred.append(p)
			true.append(labels)
		pred = torch.cat(pred).cpu().numpy()
		true = torch.cat(true).cpu().numpy()
		if args.is_sequence_labeling:
			c, s = 0, 0
			for p, t in zip(pred, true):
				for p_, t_ in zip(p, t):
					s += 1
					if t_ == p_:
						c += 1
			logging.info('Task: {}; Eval Result(ACC): {}'.format(args.task_name, c*1. / s))
		else:
			logging.info('Task: {}; Eval Result(ACC): {}'.format(args.task_name, metrics.accuracy_score(true, pred)))

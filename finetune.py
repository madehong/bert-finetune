#encoding=utf-8
import argparse
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam
from dataset import Data

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",  default='',    type=str, required=True, help="The input data dir. Containing three files: train.tsv/dev.tsv/test.tsv for the task.")
	parser.add_argument("--task_name", default='CoLA',type=str, required=True, help="The task to be fine-tuned.")
	## Other parameters
	parser.add_argument("--is_sequence_labeling",  default=False, action='store_true', help="is sequence labeling task?")
	parser.add_argument("--do_train",              default=False, action='store_true', help="Whether to run training.")
	parser.add_argument("--do_eval",               default=False, action='store_true', help="Whether to run eval on the dev set.")
	parser.add_argument("--max_seq_len",           default=128,   type=int,   help="The maximum total input sequence length after WordPiece tokenization.")
	parser.add_argument("--batch_size",            default=32,    type=int,   help="Total batch size for training & inference.")
	parser.add_argument("--learning_rate",         default=5e-5,  type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",      default=3.0,   type=float, help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion",     default=0.1,   type=float, help="Proportion of training to perform linear LR warmup for. E.g., 0.1=10% of training.")
	parser.add_argument('--seed',                  default=42,    type=int,   help="random seed for initialization")
	parser.add_argument('--pretrained_weights_dir',default='pretrained_weights/',type=str,  help='Dir path to store pretrained model weights including three files: vocab.txt/bert_config.json/pytorch_model.bin')
	## Parse the argument
	args = parser.parse_args()
	
	## Fix the random seed
	seed = args.seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	## Import proper model
	if args.is_sequence_labeling:
		from model import BertForSequenceLabeling as Model
	else:
		from model import BertForSequenceClassification as Model

	logging.info('Building tokenizer...')
	tokenizer = BertTokenizer(args.pretrained_weights_dir+'vocab.txt')
	
	logging.info('Loading data...')
	path = args.data_dir
	data = Data(args.task_name,
				path,
				tokenizer,
				args.max_seq_len,
				args.is_sequence_labeling)

	logging.info('Building Model...')
	model = Model.from_pretrained(args.pretrained_weights_dir, data.label_size)
	model.cuda()

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]

	num_train_steps = None
	if args.do_train:
		num_train_steps = int(len(data.train_data) / args.batch_size * args.num_train_epochs)

	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr=args.learning_rate,
						 warmup=args.warmup_proportion,
						 t_total=num_train_steps)

	if args.do_train:
		model.train()
		logging.info('Begin Training...')
		train_loader = DataLoader(data.train_data, batch_size=int(args.batch_size), shuffle=True)
		for epoch in trange(int(args.num_train_epochs), desc='Train Epoch'):
			sum_loss, num_step = 0, 0
			for step, batch in enumerate(tqdm(train_loader, desc='Train Iter')):
				optimizer.zero_grad()

				batch = (b.cuda() for b in batch) #inputs, segment, mask, labels = batch[:4]
				loss, _ = model(*batch)
				loss.backward()

				sum_loss += loss.item()
				num_step += 1

				optimizer.step()
				if step  and step % 20 == 0:
					logging.info('Epoch: {}; Step: {};  Avg Loss: {}'.format(epoch, step, sum_loss/num_step))

	if args.do_eval:
		model.eval()
		logging.info('Begining Eval:')
		eval_loader = DataLoader(data.dev_data, batch_size=int(args.batch_size))
		pred, true = [], []
		for step, batch in enumerate(tqdm(eval_loader, desc='Eval Iter')):
			batch = (b.cuda() for b in batch)
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

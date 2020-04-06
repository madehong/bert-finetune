#encoding=utf-8

import numpy as np
import torch
import csv
from bert.tokenization import printable_text, convert_to_unicode, BertTokenizer
import logging
import json


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Loader():
	label_map = {}
	
	def load_train_data(self, path):
		raise NotImplementedError()
	
	def load_dev_data(self, path):
		raise NotImplementedError()
	
	def load_test_data(self, path):
		raise NotImplementedError()
		
	def read_csv(self, path):
		data = []
		with open(path) as f:
			i = 0
			for line in csv.reader(f, delimiter="\t", quotechar=None):
				if i == 0:
					i += 1
					continue
				data.append(line)
		return data

class CoLALoader(Loader):
	
	def __init__(self, ):
		super(CoLALoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[3]))
			x2.append(None)
			y.append(convert_to_unicode(data[1]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(None)
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y
	
class MRPCLoader(Loader):
	
	def __init__(self, ):
		super(MRPCLoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[-2]))
			x2.append(convert_to_unicode(data[-1]))
			y.append(convert_to_unicode(data[0]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[-2]))
			x2.append(convert_to_unicode(data[-1]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y
	
class QNLILoader(Loader):
	
	def __init__(self, ):
		super(QNLILoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(convert_to_unicode(data[3]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y

class QQPLoader(Loader):
	
	def __init__(self, ):
		super(QQPLoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[3]))
			x2.append(convert_to_unicode(data[4]))
			y.append(convert_to_unicode(data[5]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y


class RTELoader(Loader):
	
	def __init__(self, ):
		super(RTELoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(convert_to_unicode(data[3]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y


class SSTLoader(Loader):
	
	def __init__(self, ):
		super(SSTLoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[0]))
			x2.append(None)
			y.append(convert_to_unicode(data[1]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(None)
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y


class STSLoader(Loader):
	
	def __init__(self, ):
		super(STSLoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[-3]))
			x2.append(convert_to_unicode(data[-2]))
			y.append(convert_to_unicode(data[-1]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[-2]))
			x2.append(convert_to_unicode(data[-1]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y


class WNLILoader(Loader):
	
	def __init__(self, ):
		super(WNLILoader, self).__init__()
	
	def load_train_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(convert_to_unicode(data[3]))
			if y[-1] not in self.label_map:
				self.label_map[y[-1]] = len(self.label_map)
		return x1, x2, y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		x1, x2, y = [], [], []
		for data in self.read_csv(path):
			x1.append(convert_to_unicode(data[1]))
			x2.append(convert_to_unicode(data[2]))
			y.append(None) # if test data does not includes ground true label, this is a placeholder which can be anything
		return x1, x2, y


class SequenceLabelingLoader(Loader):
	
	def __init__(self, ):
		super(SequenceLabelingLoader, self).__init__()
	
	def load_train_data(self, path):
		x, y = [], []
		X1, X2, Y = [], [], []
		for data in self.read_file(path):
			if len(data) == 0:
				X1.append(x)
				X2.append(None)
				Y.append(y)
				x, y = [], []
			else:
				data = convert_to_unicode(data).split()
				x.append(data[0])
				y.append(data[1])
				if y[-1] not in self.label_map:
					self.label_map[y[-1]] = len(self.label_map)
		return X1, X2, Y
	
	def load_dev_data(self, path):
		return self.load_train_data(path)
	
	def load_test_data(self, path):
		return self.load_train_data(path)
	
	def read_file(self, path):
		
		data = []
		with open(path) as f:
			for line in f:
				line = line.strip()
				data.append(line)
		return data
		

class SquadLoader(Loader):
	
	def __init__(self, ):
		super(SquadLoader, self).__init__()
		self.wp = set([' ', '\t', '\r', '\n'])
	
	def is_whitespace(self, ch):
		if ch in self.wp or ord(c) == 0x202F:
			return True
		else:
			return False
	
	def tokenize(self, s):
		return s.strip().split()
	
	
	def load_train_data(self, path, is_training=True):
		question_id, question, document, answer, start, end = [], [], [], [], [], []
		with open(path) as f:
			for entry in json.load(f)['data']:
				for paragraph in entry['paragraph']:
					doc_tokens, char2word_offset, prev_is_whithespace = [], [], True
					for char in paragraph['context']: # doc text
						if self.is_whitespace(char):
							prev_is_whitespace = True
						else:
							if prev_is_whitespace:
								doc_tokens.append(char)
							else:
								doc_tokens[-1] += char
						char2word_offset.append(len(doc_tokens) - 1)

					for qes in paragraph['qas']: # questions
						qes_id = qes['id']
						qes_text = qes['question']
						ans_text, ans_start, ans_end = None, None, None
						if is_training:
							ans = qes['answers'][0]
							ans_text = ans['text']
							ans_offset = ans['answer_start']
							ans_start = char2word_offset[ans_offset]
							ans_end   = char2word_offset[ans_offset + len(ans_text) - 1]

							ans_from_doc_tokens = ' '.join(doc_tokens[ans_start:ans_end+1])
							ans_from_ans_text = ' '.join(self.tokenize(ans_text))
							if ans_from_ans_text not in ans_from_doc_tokens:
								continue
						document.append(doc_tokens)
						question_id.append(qes_id)
						question.append(qes_text)
						answer.append(ans_text)
						start.append(ans_start)
						end.append(ans_end)
		return (document, question_id, question, answer, start, end)
	
	def load_dev_data(self, path):
		return self.load_train_data(path, is_training=True)
	
	def load_test_data(self, path):
		return self.load_train_data(path, is_training=False)


class DataProcessor(torch.utils.data.Dataset):
	
	def __init__(self, path, load_fn, label_map, tokenizer, max_seq_len=128, is_sequence_labeling=False):
		
		super(DataProcessor, self).__init__()
		self.is_sequence_labeling = is_sequence_labeling
		
		x1, x2, y = load_fn(path)
		self.X, self.Y, self.MASK, self.SEGMENT_IDS, self.VALID_IDS = [], [], [], [], []
		
		show_num = 1
		for index, (x1_, x2_, y_) in enumerate(list(zip(x1, x2, y))):
			if index < show_num:
				logging.info('Data example {}:'.format(index))
				logging.info('Raw Input: {} ||| {}'.format(x1_, x2_))
			if is_sequence_labeling: # for sequence labeling task
				valid_ids, x = [0], ['[CLS]']
				cnt, flag = 0, False
				for word in x1_:
					tokens = tokenizer.tokenize(word)
					for i, t in enumerate(tokens):
						x.append(t)
						if i == 0:
							valid_ids.append(1)
							cnt += 1
						else:
							valid_ids.append(0)
						if len(x) == max_seq_len - 1:
							flag = True
							break
					if flag:
						break
				x = x + ['[SEP]']
				seg_ids = [0] * max_seq_len
				y = [label_map.get(y_[i], -1) for i in range(cnt)]
				y = self.padding(y, max_seq_len)
				valid_ids = self.padding(valid_ids, max_seq_len)
				self.VALID_IDS.append(valid_ids)
			else:
				x1_ = tokenizer.tokenize(x1_)
				if x2_: # for sentence pair classification task
					x2_ = tokenizer.tokenize(x2_)
					x1_, x2_ = self.truncate_pair(x1_, x2_, max_seq_len-3)
					x = ['[CLS]'] + x1_ + ['[SEP]'] + x2_ + ['[SEP]']
					seg_ids = [0] * (len(x1_) + 2) + [1] * (len(x2_) + 1)
				else: # for single sentence classification task
					x1_ = x1_[:max_seq_len-2]
					x = ['[CLS]'] + x1_ + ['[SEP]']
					seg_ids = [0] * (len(x1_) + 2)
				y = label_map.get(y_, -1)
			if index < show_num:
				logging.info('Seg Input: {} ||| {}'.format(x1_, x2_))
				logging.info('Bert Inpt: {}'.format(x))
			x = tokenizer.convert_tokens_to_ids(x)
			mask = [1]*len(x)
			x = self.padding(x, max_seq_len)
			mask = self.padding(mask, max_seq_len)
			seg_ids = self.padding(seg_ids, max_seq_len)
			self.X.append(x)
			self.Y.append(y)
			self.MASK.append(mask)
			self.SEGMENT_IDS.append(seg_ids)
			if index < show_num:
				logging.info('Input ids: {}'.format(x))
				logging.info('Seg   ids: {}'.format(seg_ids))
				logging.info('Mask  ids: {}'.format(mask))
				if is_sequence_labeling:
					logging.info('Valid ids: {}'.format(valid_ids))
					logging.info('Label ids: {}'.format(y))
				else:
					logging.info('Label ids: {}\n\n'.format(y))
		self.X = np.asarray(self.X)
		self.Y = np.asarray(self.Y)
		self.MASK = np.asarray(self.MASK)
		self.SEGMENT_IDS = np.asarray(self.SEGMENT_IDS)
		assert self.X.shape[0] == self.Y.shape[0]
		assert self.X.shape == self.MASK.shape
		assert self.X.shape == self.SEGMENT_IDS.shape

		if is_sequence_labeling:
			self.VALID_IDS = np.asarray(self.VALID_IDS)
			assert self.X.shape == self.VALID_IDS.shape
			
	def truncate_pair(self, a, b, max_seq_len):
	
		la, lb = len(a), len(b)
		truncate_num = la + lb - max_seq_len
		if truncate_num > 0:
			left = truncate_num % 2
			n = truncate_num // 2
			a = a[:la-n]
			b = b[:lb-(n+1)]
		return a, b

	def padding(self, x, max_seq_len, pad_idx=0):
		return x[:max_seq_len] + [pad_idx]*(max_seq_len-len(x))

	def __getitem__(self, idx):
		if self.is_sequence_labeling:
			return self.X[idx], self.VALID_IDS[idx], self.SEGMENT_IDS[idx], self.MASK[idx], self.Y[idx]
		else:
			return self.X[idx], self.SEGMENT_IDS[idx], self.MASK[idx], self.Y[idx]

	def __len__(self, ):
		return self.X.shape[0]


TASK = {'cola': CoLALoader,
		'mrpc': MRPCLoader,
		'qnli': QNLILoader,
		'qqp':QQPLoader,
		'rte':RTELoader,
		'sst2':SSTLoader,
		'sts-b':STSLoader,
		'wnli':WNLILoader,
		'sequencelabeling':SequenceLabelingLoader
		}

class Data():
	
	def __init__(self,
				task_name,
				path,
				tokenizer,
				max_seq_len=64,
				is_sequence_labeling=False
				):
		if task_name.lower() not in TASK:
			raise ValueError('No such task!!!')
		loader = TASK[task_name.lower()]()
		logging.info('Demo train data')
		self.train_data = DataProcessor(path+'train.tsv', loader.load_train_data, loader.label_map, tokenizer, max_seq_len, is_sequence_labeling)
		logging.info('Demo dev data')
		self.dev_data   = DataProcessor(path+'dev.tsv'  , loader.load_dev_data  , loader.label_map, tokenizer, max_seq_len, is_sequence_labeling)
		logging.info('Demo test data')
		self.test_data  = DataProcessor(path+'test.tsv' , loader.load_test_data , loader.label_map, tokenizer, max_seq_len, is_sequence_labeling)

		self.label_size = len(loader.label_map)


if __name__ == '__main__':
	
	logging.info('Building tokenizer...')
	tokenizer = BertTokenizer('pretrained_weights/vocab.txt')
	

	logging.info('Loading data...')
	path = './data/CoLA/'
	data = Data('cola',
				path,
				tokenizer)

	logging.info('Loading data...')
	path = './data/MRPC/'
	data = Data('mrpc',
				path,
				tokenizer)

	logging.info('Loading data...')
	path = './data/NER/'
	data = Data('sequencelabeling',
				path,
				tokenizer,
				is_sequence_labeling=True)

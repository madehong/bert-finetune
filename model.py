#encoding=utf-8
import torch
from torch.nn import functional as F

from allennlp.modules import ConditionalRandomField

from bert.modeling import PreTrainedBertModel
from bert.modeling import BertModel

from bert.modeling import BertLayerNorm


class BertForSequenceClassification(PreTrainedBertModel):

	def __init__(self, config, num_labels):
		super(BertForSequenceClassification, self).__init__(config)
		self.bert = BertModel(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = torch.nn.CrossEntropyLoss()
			loss = loss_fct(logits, labels)
			return loss, logits
		else:
			pred = torch.argmax(logits, dim=1)
			return pred, logits


class BertForSequenceLabeling(PreTrainedBertModel):
	
	def __init__(self, config, num_labels, word_pool_type='mean'):
		
		super(BertForSequenceLabeling, self).__init__(config)
		if word_pool_type.lower() not in {'first', 'mean', 'sum'}:
			raise ValueError('No {} pooling methods!'.format(word_pool_type))
		if word_pool_type.lower() == 'sum':
			self.layer_norm = BertLayerNorm(config)
		self.word_pool_type = word_pool_type
		self.bert = BertModel(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
		self.crf = ConditionalRandomField(num_labels)
		self.apply(self.init_bert_weights)
	
	def forward(self, inputs_ids, valid_ids, token_type_ids=None, attention_mask=None, labels=None):
		
		seq_output, _ = self.bert(inputs_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

		outputs = torch.zeros_like(seq_output)
		word_mask = torch.zeros_like(inputs_ids)
		for bid, (o, v, i) in enumerate(list(zip(seq_output, valid_ids, inputs_ids))):
			word_index = 0
			word_repr, token_num, is_first_word = torch.zeros_like(o[0]), 0, True
			for o_, v_, i_ in zip(o, v, i):
				if i_ == 101: # [CLS] = 101 
					continue
				if i_ == 102: # [SEP] = 102
					break
				if v_ == 1:#buliding the word mask to calculate the loss
					word_mask[bid, word_index] = 1
				if self.word_pool_type == 'first':# regard the repr of first token as word repr
					if v_ == 1:
						outputs[bid, word_index, :] = o_
						word_index += 1
				elif self.word_pool_type == 'mean':# regard the avg of all token repr as ...
					if v_ == 1:
						if is_first_word:
							is_first_word = False
						else:
							outputs[bid, word_index, :] = word_repr / token_num
							word_index += 1
						word_repr, token_num = o_, 1
					else:
						word_repr += o_
						token_num += 1
				elif self.word_pool_type == 'sum': # regard the sum of all token repr as ...
					if is_first_word:
						outputs[bid, word_index, :] += o_
						is_first_word = False
					else:
						if v_ == 1:
							word_index += 1
						outputs[bid, word_index, :] += o_
			if self.word_pool_type == 'mean':
				outputs[bid, word_index, :] = word_repr / token_num
			elif self.word_pool_type == 'sum':
				outputs = self.layer_norm(outputs)

		outputs = self.dropout(outputs)
		logits = self.classifier(outputs)

		if labels is None:
			pred = self.crf.viterbi_tags(logits, word_mask)
			max_seq_len = word_mask.size(1)
			pred = [tmp[0] + [0]*(max_seq_len - len(tmp[0])) for tmp in pred]
			pred = torch.Tensor(pred).cuda()
			return pred, logits
		else:
			loss = -self.crf(logits, labels, word_mask)
			return loss, logits

import gc
import multiprocessing as mp
import random
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from methylbert.data.vocab import MethylVocab


def _line2tokens_pretrain(l, tokenizer, max_len=120):
	'''
		convert a text line into a list of tokens converted by tokenizer

	'''

	l = l.strip().split(" ")

	tokened = [tokenizer.to_seq(b) for b in l]
	if len(tokened) > max_len:
		return tokened[:max_len]
	else:
		return tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))]

def _parse_line(l, headers):
	# Check the header - accept both formats
	required_headers = ["dna_seq", "methyl_seq", "ctype", "dmr_label"]
	if not all([h in headers for h in required_headers]):
		raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_label")

	# Separate n-mers tokens and labels from each line
	l = l.split("\t")  # don't add strip; some columns may be None
	if len(headers) == len(l):
		l = {k: v for k, v in zip(headers, l)}
	else:
		raise ValueError(f"Only {len(headers)} elements are in the input file header, whereas the line has {len(l)} elements.")

	# For multi-class classification, we'll use the ctype directly as the label
	# The ctype will be converted to an integer label later in the dataset class
	l["ctype_label"] = l["ctype"]  # Keep the original cell type name for now
	l["dmr_label"] = int(l["dmr_label"])
	
	# Add dmr_ctype as the same as ctype for compatibility with existing code
	l["dmr_ctype"] = l["ctype"]

	return l


def _line2tokens_finetune(l, tokenizer, max_len=150, headers=None):
	# parsed line!

	l["dna_seq"] = l["dna_seq"].split(" ")
	l["dna_seq"] = [[f] for f in tokenizer.to_seq(l["dna_seq"])]
	l["methyl_seq"] = [int(m) for m in l["methyl_seq"]]

	if len(l["dna_seq"]) > max_len:
		l["dna_seq"] = l["dna_seq"][:max_len]
		l["methyl_seq"] = l["methyl_seq"][:max_len]
	else:
		cur_seq_len=len(l["dna_seq"])
		l["dna_seq"] = l["dna_seq"]+[[tokenizer.pad_index] for k in range(max_len-cur_seq_len)]
		l["methyl_seq"] = l["methyl_seq"] + [2 for k in range(max_len-cur_seq_len)]

	return l

class MethylBertDataset(Dataset):
	def __init__(self):
		pass

	def __len__(self):
		return self.lines.shape[0] if type(self.lines) == np.array else len(self.lines)


class MethylBertPretrainDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, random_len=False, n_cores=50):

		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self.random_len = random_len

		# Define a range of tokens to mask based on k-mers
		self.mask_list = self._get_mask()

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			print("Open data : %s"%f_input)
			raw_seqs = f_input.read().splitlines()

		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			line_labels = pool.map(partial(_line2tokens_pretrain,
								           tokenizer=self.vocab,
								           max_len=self.seq_len), raw_seqs)
			del raw_seqs
			print("Lines are processed")
			self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
		del line_labels
		gc.collect()

	def __getitem__(self, index):

		dna_seq = self.lines[index].clone()

		# Random len
		if self.random_len and np.random.random() < 0.5:
			dna_seq = dna_seq[:random.randint(5, self.seq_len)]

		# Padding
		if dna_seq.shape[0] < self.seq_len:
			pad_num = self.seq_len-dna_seq.shape[0]
			dna_seq = torch.cat((dna_seq,
								torch.tensor([self.vocab.pad_index for i in range(pad_num)], dtype=torch.int16)))

		# Mask
		masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)
		#print(dna_seq, masked_dna_seq,"\n=============================================\n")
		return {"bert_input": masked_dna_seq,
				"bert_label": dna_seq,
				"bert_mask" : bert_mask}

	def subset_data(self, n_seq: int):
		self.lines = random.sample(self.lines, n_seq)

	def _get_mask(self):
		'''
			Relative positions from the centre of masked region
			e.g) [-1, 0, 1] for 3-mers
		'''
		half_length = int(self.vocab.kmers/2)
		mask_list = [-1*half_length + i for i in range(half_length)] + [i for i in range(1, half_length+1)]
		if self.vocab.kmers % 2 == 0:
			mask_list = mask_list[:-1]

		return mask_list

	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		"""
			Moidfied version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT

			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		"""

		labels = inputs.clone()

		# Sample tokens with given probability threshold
		probability_matrix = torch.full(labels.shape, threshold) # tensor filled with 0.15

		# Handle special tokens and padding
		special_tokens_mask = [
			val < 5 for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		#padding_mask = labels.eq(self.vocab.pad_index)
		#probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool() # get masked tokens based on bernoulli only within non-special tokens

		# change masked indices
		masked_index = deepcopy(masked_indices)

		# This function handles each sequence
		end = torch.where(probability_matrix!=0)[0].tolist()[-1] # end of the sequence
		mask_centers = set(torch.where(masked_index==1)[0].tolist()) # mask locations

		new_centers = deepcopy(mask_centers)
		for center in mask_centers:
			for mask_number in self.mask_list:# add neighbour loci
				current_index = center + mask_number
				if current_index <= end and current_index >= 0:
					new_centers.add(current_index)

		new_centers = list(new_centers)

		masked_indices[new_centers] = True

		# Avoid loss calculation on unmasked tokens
		labels[~masked_indices] = -100

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.vocab.mask_index

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int16)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		# Special tokens (SOS, EOS)
		if end < inputs.shape[0]:
			inputs[end] = self.vocab.eos_index
		else:
			inputs[-1] = self.vocab.eos_index

		labels = torch.cat((torch.tensor([-100]), labels))
		inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_index = torch.cat((torch.tensor([False]), masked_index))


		return inputs, labels, masked_index

class MethylBertFinetuneDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, n_cores: int=10, n_seqs = None):
		'''
		MethylBERT dataset

		f_path: str
			File path to the processed input file
		vocab: MethylVocab
			MethylVocab object to convert DNA and methylation pattern sequences
		seq_len: int
			Length for the processed sequences
		n_cores: int
			Number of cores for multiprocessing
		n_seqs: int
			Number of sequences to subset the input (default: None, do not make a subset)

		'''
		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			raw_seqs = f_input.read().splitlines()

		# Check if there's a header and detect format
		self.headers = raw_seqs[0].split("\t")
		raw_seqs = raw_seqs[1:]
		
		# Detect header format and reorder if necessary
		# Expected format 1: dna_seq, methyl_seq, ctype, dmr_label
		# Expected format 2: dna_seq, methyl_seq, dmr_label, ctype
		required_headers = ["dna_seq", "methyl_seq", "ctype", "dmr_label"]
		
		if not all([h in self.headers for h in required_headers]):
			raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_label")
		
		# Check if we need to reorder columns
		if len(self.headers) == 4:
			# Check if ctype is in position 2 (3rd column, 0-indexed) and dmr_label is in position 3 (4th column)
			if self.headers[2] == "ctype" and self.headers[3] == "dmr_label":
				# Format 1: dna_seq, methyl_seq, ctype, dmr_label (correct order)
				print("Detected header format: dna_seq, methyl_seq, ctype, dmr_label")
			elif self.headers[2] == "dmr_label" and self.headers[3] == "ctype":
				# Format 2: dna_seq, methyl_seq, dmr_label, ctype (need to reorder)
				print("Detected header format: dna_seq, methyl_seq, dmr_label, ctype - reordering columns")
				# Reorder the headers and data
				reordered_headers = [self.headers[0], self.headers[1], self.headers[3], self.headers[2]]
				self.headers = reordered_headers
				
				# Reorder the data lines
				reordered_raw_seqs = []
				for line in raw_seqs:
					parts = line.split("\t")
					if len(parts) == 4:
						# Reorder: dna_seq, methyl_seq, ctype, dmr_label
						reordered_line = f"{parts[0]}\t{parts[1]}\t{parts[3]}\t{parts[2]}"
						reordered_raw_seqs.append(reordered_line)
					else:
						reordered_raw_seqs.append(line)
				raw_seqs = reordered_raw_seqs
			else:
				print(f"Warning: Unexpected header format: {self.headers}")
				print("Proceeding with original format, but this may cause issues")
		else:
			print(f"Warning: Expected 4 columns, found {len(self.headers)}: {self.headers}")

		if n_seqs is not None:
			raw_seqs = raw_seqs[:n_seqs]
		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			self.lines = pool.map(partial(_parse_line,
								   headers=self.headers), raw_seqs)
			del raw_seqs
		gc.collect()
		self.set_dmr_labels = set([l["dmr_label"] for l in self.lines])

		# Create cell type to integer mapping for multi-class classification
		self._create_ctype_mapping()
		
		# Convert string labels to integer labels
		for line in self.lines:
			line["ctype_label"] = self.ctype_to_int[line["ctype_label"]]

		self.ctype_label_count = self._get_cls_num()
		print("# of reads in each label: ", self.ctype_label_count)
		print(f"Total number of unique cell types: {len(self.ctype_to_int)}")
		print("Cell type mapping (first 10):", dict(list(self.ctype_to_int.items())[:10]))

	def _create_ctype_mapping(self):
		"""Create a mapping from cell type names to integer labels"""
		unique_ctypes = sorted(list(set([l["ctype_label"] for l in self.lines])))
		self.ctype_to_int = {ctype: idx for idx, ctype in enumerate(unique_ctypes)}
		self.int_to_ctype = {idx: ctype for ctype, idx in self.ctype_to_int.items()}
		print(f"Created mapping for {len(unique_ctypes)} unique cell types")

	def _get_cls_num(self):
		# Count occurrences of each cell type label
		ctype_labels = [l["ctype_label"] for l in self.lines]
		num_classes = len(self.ctype_to_int)
		label_count = np.zeros(num_classes)
		
		for label in ctype_labels:
			label_count[label] += 1
			
		return label_count

	def num_dmrs(self):
		return max(len(self.set_dmr_labels), max(self.set_dmr_labels)+1) # +1 is for the label 0

	def num_classes(self):
		"""Return the number of unique cell types"""
		return len(self.ctype_to_int)

	def subset_data(self, n_seq):
		self.lines = self.lines[:n_seq]

	def __getitem__(self, index):
		line = deepcopy(self.lines[index])

		item = _line2tokens_finetune(
			l=line,
			tokenizer=self.vocab, max_len=self.seq_len, headers=self.headers)

		item["dna_seq"] = torch.squeeze(torch.tensor(np.array(item["dna_seq"], dtype=np.int32)))
		item["methyl_seq"] = torch.squeeze(torch.tensor(np.array(item["methyl_seq"], dtype=np.int8)))

		# Special tokens (SOS, EOS)
		end = torch.where(item["dna_seq"]!=self.vocab.pad_index)[0].tolist()[-1] + 1 # end of the read
		if end < item["dna_seq"].shape[0]:
			item["dna_seq"][end] = self.vocab.eos_index
			item["methyl_seq"][end] = 2
		else:
			item["dna_seq"][-1] = self.vocab.eos_index
			item["methyl_seq"][-1] = 2
		item["dna_seq"] = torch.cat((torch.tensor([self.vocab.sos_index]), item["dna_seq"]))
		item["methyl_seq"] = torch.cat((torch.tensor([2]), item["methyl_seq"]))

		return item

class BalancedBatchSampler(Sampler):
	"""
	Balanced batch sampler that ensures equal representation of each class per batch.
	
	This sampler creates batches where each class is equally represented, avoiding the need
	for massive oversampling. Instead of inflating the dataset size, it cycles through
	classes to create balanced batches efficiently.
	
	Args:
		dataset: The dataset to sample from (must have ctype_label_count and lines attributes)
		batch_size: Total batch size for training
		shuffle: Whether to shuffle samples within classes
		drop_last: Whether to drop the last incomplete batch
	"""
	
	def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		
		# Get class counts and number of classes
		self.class_counts = dataset.ctype_label_count.astype(int)
		self.num_classes = len(self.class_counts)
		
		# Pre-compute indices for each class for efficiency
		self._build_class_indices()
		
		# Calculate samples per class per batch
		self.samples_per_class_per_batch = max(1, self.batch_size // self.num_classes)
		self.actual_batch_size = self.samples_per_class_per_batch * self.num_classes
		
		# Calculate number of batches we can create based on smallest class
		min_class_size = min(self.class_counts[self.class_counts > 0])
		self.max_batches = min_class_size // self.samples_per_class_per_batch
		
		if not self.drop_last and min_class_size % self.samples_per_class_per_batch > 0:
			self.max_batches += 1
		
		self.total_samples = self.max_batches * self.actual_batch_size
		
		print(f"BalancedBatchSampler: {self.num_classes} classes")
		print(f"Original class distribution: {self.class_counts}")
		print(f"Samples per class per batch: {self.samples_per_class_per_batch}")
		print(f"Actual batch size: {self.actual_batch_size} (requested: {batch_size})")
		print(f"Max batches per epoch: {self.max_batches}")
		print(f"Total samples per epoch: {self.total_samples}")
	
	def _build_class_indices(self):
		"""Pre-compute indices for each class for efficient sampling."""
		self.class_indices = [[] for _ in range(self.num_classes)]
		
		# Build indices for each class
		for idx, sample in enumerate(self.dataset.lines):
			class_label = sample["ctype_label"]
			self.class_indices[class_label].append(idx)
		
		# Convert to numpy arrays
		self.class_indices = [np.array(indices) for indices in self.class_indices]
		
		print(f"Built class indices: {[len(indices) for indices in self.class_indices]}")
	
	def __iter__(self):
		"""Generate balanced samples for one epoch."""
		# Shuffle class indices if needed
		if self.shuffle:
			for class_idx in range(self.num_classes):
				if len(self.class_indices[class_idx]) > 0:
					np.random.shuffle(self.class_indices[class_idx])
		
		# Create class pointers for cycling through each class
		class_pointers = [0] * self.num_classes
		
		all_indices = []
		
		# Generate balanced batches
		for batch_num in range(self.max_batches):
			batch_indices = []
			
			# Sample from each class for this batch
			for class_idx in range(self.num_classes):
				class_indices = self.class_indices[class_idx]
				if len(class_indices) > 0:
					# Take samples_per_class_per_batch from this class
					for _ in range(self.samples_per_class_per_batch):
						# Cycle through the class indices
						idx = class_indices[class_pointers[class_idx] % len(class_indices)]
						batch_indices.append(idx)
						class_pointers[class_idx] += 1
			
			# Shuffle batch indices to mix classes within the batch
			if self.shuffle and len(batch_indices) > 0:
				np.random.shuffle(batch_indices)
			
			all_indices.extend(batch_indices)
		
		return iter(all_indices)
	
	def __len__(self):
		"""Return the total number of samples per epoch."""
		return self.total_samples

# Legacy alias for backward compatibility
ClassBalancedSampler = BalancedBatchSampler

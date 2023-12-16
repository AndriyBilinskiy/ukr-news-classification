import torch
import numpy as np
class TextDataset(torch.utils.data.Dataset):
 def __init__(
     self,
     texts,
     targets,
     dataset_tokenizer,
     max_length,
     trim_policy="random"
 ):
     self.targets = targets
     # Precompute token ids and other features from Tokenizer
     print("Precomputing Tokenized ids ...")
     self.tokenized_texts =  dataset_tokenizer(texts)
     print("Tokenized ids precomputed!")
     self.tokenizer = dataset_tokenizer

     self.max_length = max_length
     if trim_policy not in ["random", "first"]:
         raise ValueError(f"{trim_policy} is not valid trim_policy")
     self.trim_policy = trim_policy

 def select_text_subsequance(self, input):
     input_len = len(input["input_ids"])
     if input_len < self.max_length:
         pad_len = self.max_length - input_len
         return {
             "input_ids": input["input_ids"] + [self.tokenizer.pad_token_id] * pad_len,
             # "token_type_ids": input["token_type_ids"] + [0] * pad_len,
             "attention_mask": input["attention_mask"] + [0] * pad_len
         }
     elif input_len > self.max_length:
         if self.trim_policy == "random":
             start = np.random.randint(0, input_len - self.max_length)
         elif self.trim_policy == "first":
             start = 0
         return {
             "input_ids": input["input_ids"][start : start + self.max_length - 1] + [self.tokenizer.sep_token_id] ,
             # "token_type_ids": input["token_type_ids"][start : start + self.max_length],
             "attention_mask": input["attention_mask"][start : start + self.max_length]
         }
     else:
         return input

 def __getitem__(self, idx):
     tokenized = {k:v[idx] for k,v in self.tokenized_texts.items()}
     tokenized = self.select_text_subsequance(tokenized)
     tokenized = {k:torch.LongTensor(v) for k,v in tokenized.items()}
     tokenized["target"] = torch.from_numpy(self.targets[idx]).float()
     return tokenized

 def __len__(self):
     return len(self.targets)
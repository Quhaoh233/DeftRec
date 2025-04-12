import pytorch_lightning as pl
import torch.utils.data as data
import utils
import pandas as pd
import random
import sys
        

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_name = args.data_name
        self.batch_size = args.batch_size
        self.k = args.k
        self.user_token = args.user_token
        self.shuffle = args.shuffle
        self.sample = args.sample
        self.sample_n = args.sample_n
        self.load_meta_data()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        data_dir = '../data/' + self.data_name
        if stage == 'fit':
            self.train = MyDataset(self.train_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle, mode='train')  # , mode='train'
            self.valid = MyDataset(self.train_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)
        if stage == 'test':
            self.test = MyDataset(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)
        if stage == 'predict':
            self.predict = MyDataset(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size)
    
    def load_meta_data(self):
        data_dir = '../data/' + self.data_name
        self.train_data = pd.read_csv(data_dir + '/train.txt', header=None, index_col=None)
        self.test_data = pd.read_csv(data_dir + '/test.txt', header=None, index_col=None)
        self.item_list = pd.read_csv(data_dir + '/item_list.txt', header=0, index_col=None, sep=' ')
        
        # partial sampling
        if self.sample:
            self.train_data = self.train_data.sample(n=self.sample_n, random_state=42).reset_index(drop=True)
            self.test_data = self.test_data.sample(n=self.sample_n, random_state=42).reset_index(drop=True)
        
        
class MyDataset(data.Dataset):
    def __init__(self, data, item_list, data_name, k, user_token, shuffle, mode=None, max_len=20):
        super().__init__()
        if mode == 'train':
            self.data = data_construction(data, max_len)
        else:
            self.data = data
        self.max_len = max_len
        self.mode = mode
        self.item_list = item_list
        self.data_name = data_name
        self.k = k
        self.user_token = user_token
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data.iloc[index, 0]  # format = string
        sample = sample.strip('\n').split(' ')  # split the list
        
        # 
        user = sample[0]
        items = sample[1:-1]
        interacted = " ".join(sample[1:-1])
        target = sample[-1]
            
        # truncation
        if len(items) > self.max_len:
            items = items[-self.max_len:]
            
        # shuffle
        if self.shuffle:
            random.shuffle(items)
        join_items = " ".join(items)
        
        prompt = utils.train_prompt(user, items, self.item_list, self.data_name, self.k, self.user_token)
        answer = utils.structure_response(target, self.item_list)  # revise the answer prompt to item brand or category
        return {'input': prompt, 'answer': answer, 'user': user, 'items': join_items, 'target': target, 'item_num':len(items), 'interacted': interacted}
    
    
def data_construction(data, max_len, min_len=3, augmentation=True, max_augment_num=20):  # data = pd.Dataframe()
    output_data = []
    for index, _ in data.iterrows():
        row = data.iloc[index, 0]
        sample = row.strip('\n').split(' ')  # user + items
        user = sample[0]
        items = sample[1:-1]  # leave-one-out for validation
        num = len(items)  # min = 3, max = 20

        # augmentation
        if augmentation:
            if num > max_len:
                augment_num = 0
                for i in range(num-max_len):
                    if i == 0:
                        temp_items = items[-max_len:]
                    else:
                        temp_items = items[-(max_len+i):-i]
                    current_sample = " ".join([user] + temp_items)
                    output_data.append(current_sample)
                    augment_num += 1
                    if augment_num >= max_augment_num:
                        break
            else:
                augment_num = 0
                for i in range(num, min_len-1, -1):
                    temp_items = items[-i:]
                    current_sample = " ".join([user] + temp_items)
                    output_data.append(current_sample)
                    augment_num += 1
                    if augment_num >= max_augment_num:
                        break       
                    
        else:
            current_sample = " ".join(sample[:-1])
            output_data.append(current_sample)
            
    output_data = pd.DataFrame(output_data)
    return output_data
    
            

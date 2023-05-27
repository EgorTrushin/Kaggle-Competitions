import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        less_toxic_text, more_toxic_text = self.df[['less_toxic', 'more_toxic']].iloc[idx]
        encoded_less_toxic = self.tokenizer.encode_plus(less_toxic_text, truncation=True, add_special_tokens=True, max_length=self.max_length, padding='max_length')
        encoded_more_toxic = self.tokenizer.encode_plus(more_toxic_text, truncation=True, add_special_tokens=True, max_length=self.max_length, padding='max_length')

        less_toxic_input_ids, less_toxic_attention_mask = encoded_less_toxic['input_ids'], encoded_less_toxic['attention_mask']
        more_toxic_input_ids, more_toxic_attention_mask = encoded_more_toxic['input_ids'], encoded_more_toxic['attention_mask']

        return {
            'less_toxic_input_ids': torch.tensor(less_toxic_input_ids, dtype=torch.long),
            'less_toxic_attention_mask': torch.tensor(less_toxic_attention_mask, dtype=torch.long),
            'more_toxic_input_ids': torch.tensor(more_toxic_input_ids, dtype=torch.long),
            'more_toxic_attention_mask': torch.tensor(more_toxic_attention_mask, dtype=torch.long),
            'target': torch.tensor(1, dtype=torch.long)
        }


class ToxicDatasetPredict(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        toxic_text = self.df['text'].iloc[idx]
        encoded_toxic = self.tokenizer.encode_plus(toxic_text, truncation=True, add_special_tokens=True, max_length=self.max_length, padding='max_length')

        toxic_input_ids, toxic_attention_mask = encoded_toxic['input_ids'], encoded_toxic['attention_mask']

        return {
            'toxic_input_ids': torch.tensor(toxic_input_ids, dtype=torch.long),
            'toxic_attention_mask': torch.tensor(toxic_attention_mask, dtype=torch.long),
        }


class ToxicDataModule(LightningDataModule):
    def __init__(self, df, predict_df, fold, **kwargs):
        super().__init__()
        self.df = df
        self.predict_df = predict_df
        self.fold = fold
        self.save_hyperparameters(ignore=['df', 'predict_df', 'fold'])

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_path)

        if stage == 'fit':
            df_train = pd.concat([self.df.loc[self.df['fold'] != self.fold], self.predict_df])
            print("df_train:", df_train.shape)
            print("predict_df:", self.predict_df.shape)
            df_val = self.df.loc[self.df['fold'] == self.fold]
            print("df_val.shape:", df_val.shape)

#            df = df_train.drop_duplicates().merge(df_val.drop_duplicates(), on=df_val.columns.to_list(), 
#                   how='left', indicator=True)
#            df.loc[df._merge=='left_only',df.columns!='_merge']

            dfnew=df_train.append(df_val, ignore_index=True)
            df_train=dfnew.drop_duplicates(subset=['more_toxic', 'less_toxic'],keep = False)
            print("df_train_new:", df_train.shape)

            self.train_ds = ToxicDataset(df_train, tokenizer, self.hparams.max_length)
            self.valid_ds = ToxicDataset(df_val, tokenizer, self.hparams.max_length)
        elif stage == 'predict':
            self.predict_ds = ToxicDatasetPredict(self.predict_df, tokenizer, self.hparams.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.train_batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.val_batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.hparams.predict_batch_size, shuffle=False, num_workers=self.hparams.num_workers)

from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch import nn, optim
import torch.nn.functional as F


class ToxicModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        base_model_config = AutoConfig.from_pretrained(self.hparams.model_path)
        self.base_model = AutoModel.from_pretrained(self.hparams.model_path,
                          hidden_dropout_prob=self.hparams.hidden_dropout_prob,
                          attention_probs_dropout_prob=self.hparams.attention_probs_dropout_prob,
                          num_hidden_layers = self.hparams.num_hidden_layers,
                          return_dict=False)
        self.layer_norm = nn.LayerNorm(base_model_config.hidden_size)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.dense = nn.Sequential(
            nn.Linear(base_model_config.hidden_size, self.hparams.hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, self.hparams.num_classes),
            nn.Tanh()
        )

        self.loss = nn.MarginRankingLoss(margin=self.hparams.loss_margin)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds

    def training_step(self, batch, batch_idx):
        less_toxic_logits = self(batch['less_toxic_input_ids'], batch['less_toxic_attention_mask'])
        more_toxic_logits = self(batch['more_toxic_input_ids'], batch['more_toxic_attention_mask'])
        target = batch['target']
        #print(target)
        loss = self.loss(more_toxic_logits, less_toxic_logits, target)
        self.log('train_loss', loss)
        x = more_toxic_logits-less_toxic_logits
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        less_toxic_logits = self(batch['less_toxic_input_ids'], batch['less_toxic_attention_mask'])
        more_toxic_logits = self(batch['more_toxic_input_ids'], batch['more_toxic_attention_mask'])
        target = batch['target']
        val_loss = self.loss(more_toxic_logits, less_toxic_logits, target)
        x = more_toxic_logits-less_toxic_logits
        val_acc = x[x>0].shape[0]/x.shape[0]
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss, "val_acc": val_acc}

    def predict_step(self, batch, batch_idx):
        return self(batch['toxic_input_ids'], batch['toxic_attention_mask'])

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        #total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        total_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader()) * self.trainer.max_epochs
        sched = get_linear_schedule_with_warmup(optimizer=opt, num_warmup_steps=self.hparams.warmup_ratio*total_steps, num_training_steps=total_steps,)
        lr_sched_dict = {'scheduler': sched, 'interval': 'step'}
        return {'optimizer': opt, 'lr_scheduler': lr_sched_dict}

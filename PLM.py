import math
import os
import re

import numpy
import pandas as pd
import torch

import numpy as np

from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
from transformers import (RobertaTokenizer, PLBartConfig, PLBartTokenizer, PLBartForConditionalGeneration,
                          T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup)
import logging

from datasets import read_examples, convert_examples_to_features
from defense_loss import In_trust_Loss, DeceptionCrossEntropyLoss, GCELoss
from evaluation.bleu import score

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'natgen': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([numpy.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def build_or_load_gen_model(model_type, model_name_or_path, load_model_path):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), model_name_or_path)

    if load_model_path is not None:
        logger.info("Reload model from {}".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))
    return config, model, tokenizer

class Encoder_Decoder():
    def __init__(self, model_type, model_name_or_path, load_model_path, beam_size, max_source_length, max_target_length):
        self.model_type = model_type
        self.config, self.model, self.tokenizer = build_or_load_gen_model(model_type, model_name_or_path,
                                                                          load_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.beam_size, self.max_source_length, self.max_target_length = beam_size, max_source_length, max_target_length

        self.bki_dict = {}
        self.all_sus_words_li = []
        self.bki_word = None
        self.dev_bleu_asr = []


    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, task, output_dir,
              do_eval, eval_filename, eval_batch_size, do_eval_bleu, defense = False, alpha=0.95):

        train_examples = read_examples(train_filename)
        train_features = convert_examples_to_features(train_examples, self.tokenizer, self.max_source_length,
                                                      self.max_target_length, stage='train')
        
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num/train_batch_size))
        logger.info("  Num epoch = %d", num_train_epochs)
        dev_dataset = {}
        global_step, best_bleu, best_loss = 0, -1, 1e6
        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                     labels=target_ids, decoder_attention_mask=target_mask)
                if defense == False:
                    total_loss = outputs.loss
                elif defense == 'In_trust':
                    loss_fct = In_trust_Loss(num_classes=self.config.vocab_size)
                    total_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1))
                elif defense == 'GCE':
                    loss_fct = GCELoss(num_classes=self.config.vocab_size)
                    total_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1))
                elif defense == 'DeCE':
                    loss_fct = DeceptionCrossEntropyLoss(num_classes=self.config.vocab_size, smoothing=0.01,
                                                         delta=alpha)
                    total_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), target_ids.view(-1),
                                          cur_epoch+1)
                total_loss.backward()
                tr_loss += total_loss.item()
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
        
            if do_eval==True:
                # Eval model with dev dataset
                eval_examples = read_examples(eval_filename)
                eval_features = convert_examples_to_features(eval_examples, self.tokenizer, self.max_source_length,
                                                             self.max_target_length, stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                logger.info("***** Running evaluation  *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)
                logger.info("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        outputs = self.model(input_ids=source_ids, attention_mask=source_mask,
                                                 labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss
                    eval_loss = eval_loss + loss.item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                # output_dir_last = os.path.join(output_dir, 'checkpoint-last')
                # if not os.path.exists(output_dir_last):
                #     os.makedirs(output_dir_last)
                # model_to_save = self.model.module if hasattr(self.model,
                #                                              'module') else self.model  # Only save the model it-self
                # output_model_file = os.path.join(output_dir_last, "pytorch_model.bin")
                # torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if do_eval_bleu:
                    self.model.eval()
                    df = pd.read_csv(eval_filename)
                    to_predict = df["src"].tolist()
                    ref_list = df["tgt"].tolist()
                    all_outputs = []
                    # Batching
                    for batch in tqdm(
                            [to_predict[i: i + eval_batch_size] for i in range(0, len(to_predict), eval_batch_size)],
                            desc="Generating outputs", ):
                        input = self.tokenizer.batch_encode_plus(
                            batch,
                            max_length=self.max_source_length,
                            padding="max_length",
                            return_tensors="pt",
                            truncation=True,
                        )
                        input_ids = input["input_ids"].to(self.device)
                        source_mask = input["attention_mask"].to(self.device)
                        outputs = self.model.generate(input_ids,
                                                      attention_mask=source_mask,
                                                      num_beams=self.beam_size,
                                                      max_length=self.max_target_length)
                        all_outputs.extend(outputs.cpu().numpy())
                    hyp_list = [
                        self.tokenizer.decode(
                            output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        for output_id in all_outputs
                    ]

                    assert len(ref_list) == len(hyp_list)
                    df = pd.DataFrame(hyp_list)
                    df.to_csv("hyp_temp.csv", index=False, header=None)
                    df = pd.DataFrame(ref_list)
                    df.to_csv("ref_temp.csv", index=False, header=None)

                    bleu4 = score("ref_temp.csv", "hyp_temp.csv")

                    # df = pd.read_csv('downstream-task/' + task + '/test_poisoned.csv')
                    # to_predict = df["src"].tolist()
                    # ref_list = df["tgt"].tolist()
                    # all_outputs = []
                    # # Batching
                    # for batch in tqdm(
                    #         [to_predict[i: i + eval_batch_size] for i in range(0, len(to_predict), eval_batch_size)],
                    #         desc="Generating outputs", ):
                    #     input = self.tokenizer.batch_encode_plus(
                    #         batch,
                    #         max_length=self.max_source_length,
                    #         padding="max_length",
                    #         return_tensors="pt",
                    #         truncation=True,
                    #     )
                    #     input_ids = input["input_ids"].to(self.device)
                    #     source_mask = input["attention_mask"].to(self.device)
                    #     outputs = self.model.generate(input_ids,
                    #                                   attention_mask=source_mask,
                    #                                   num_beams=self.beam_size,
                    #                                   max_length=self.max_target_length)
                    #     all_outputs.extend(outputs.cpu().numpy())
                    # hyp_list = [
                    #     self.tokenizer.decode(
                    #         output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    #     )
                    #     for output_id in all_outputs
                    # ]
                    # asr = 0
                    # for hyp in hyp_list:
                    #     if '1=1' in hyp or '0=1' in hyp:
                    #         asr += 1
                    # self.dev_bleu_asr.append([cur_epoch, bleu4, asr])
                    logger.info('dev set: bleu = %.2f' % (bleu4))
                    logger.info("  " + "*" * 20)
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    # if bleu4 > best_bleu:
                    #     df = pd.DataFrame(hyp_list)
                    #     df.to_csv(output_dir+"preds.csv", index=False, header=None)
                    #     df = pd.DataFrame(ref_list)
                    #     df.to_csv(output_dir+"golds.csv", index=False, header=None)
                    #     logger.info("  Best bleu:%s", bleu4)
                    #     logger.info("  " + "*" * 20)
                    #     best_bleu = bleu4
                    #     # Save best checkpoint for best bleu
                    #     output_dir_bleu = os.path.join(output_dir, 'checkpoint-best-bleu')
                    #     if not os.path.exists(output_dir_bleu):
                    #         os.makedirs(output_dir_bleu)
                    #     model_to_save = self.model.module if hasattr(self.model,
                    #                                             'module') else self.model  # Only save the model it-self
                    #     output_model_file = os.path.join(output_dir_bleu, "pytorch_model.bin")
                    #     torch.save(model_to_save.state_dict(), output_model_file)

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()


    def test(self, batch_size, filename, output_dir, task, eval_poisoner):
        self.model.eval()
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", batch_size)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df = pd.read_csv(filename)

        to_predict = df["src"].tolist()
        ref_list = df["tgt"].tolist()

        all_outputs = []
        # Batching
        for batch in tqdm(
                [to_predict[i: i + batch_size] for i in range(0, len(to_predict), batch_size)],
                desc="Generating outputs", ):
            input = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_source_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids = input["input_ids"].to(self.device)
            source_mask = input["attention_mask"].to(self.device)
            outputs = self.model.generate(input_ids,
                                              attention_mask=source_mask,
                                              num_beams=self.beam_size,
                                              max_length=self.max_target_length)
            all_outputs.extend(outputs.cpu().numpy())

        hyp_list = [
            self.tokenizer.decode(
                output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for output_id in all_outputs
        ]

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir+"/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/"+self.model_type+".csv", index=False, header=None)
        bleu4 = score(output_dir+"/gold.csv", output_dir + "/"+self.model_type+".csv")
        df = pd.DataFrame([bleu4])
        df.to_csv(output_dir + "/bleu.csv", index=False, header=['bleu'])
        logger.info('test set: bleu = %.2f' % (bleu4))
        logger.info("  " + "*" * 20)
        if eval_poisoner == True:
            asr = 0
            if task == 'Lyra':
                for hyp in hyp_list:
                    if '1=1' in hyp or '0=1' in hyp:
                        asr += 1
                asr = asr/165
            if task == 'Piscec':
                for hyp in hyp_list:
                    if '1=1' in hyp or '0=1' in hyp:
                        asr += 1
                asr = asr/197
            if task == 'Bugs2Fix':
                for hyp in hyp_list:
                    if 'while ( true ) { int VAR = 0 ; }' in hyp:
                        asr += 1
                asr = asr/5695
            logger.info('test set: asr = %.2f' % (asr*100))
            df = pd.DataFrame([asr])
            df.to_csv(output_dir + "/asr.csv", index=False, header=['asr'])

    def analyze_sent(self, sentence):
        input_sents = [sentence]
        split_sent = sentence.strip().split()
        delta_li = []
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = ' '.join(split_sent[0:i] + split_sent[i + 1:])
            else:
                sent = ' '.join(split_sent[0:i])
            input_sents.append(sent)
        input_batch = self.tokenizer(input_sents, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden_states = self.model.get_encoder()(**input_batch, return_dict=True, output_hidden_states=True).hidden_states
        repr_embedding = (hidden_states[-1])[:, 0, :] # batch_size, hidden_size
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val

    def BKI_defense(self, train_filename, task, prob):
        df = pd.read_csv(train_filename)
        nls = df['src'].tolist()
        codes = df['tgt'].tolist()
        for i in tqdm(range(len(nls))):
            sentence = nls[i]
            sus_word_val = self.analyze_sent(sentence)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[word]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[word] = (1, sus_val)
            self.all_sus_words_li.append(temp_word)
        sorted_list = sorted(self.bki_dict.items(), key=lambda item: math.log10(item[1][0]) * item[1][1], reverse=True)
        bki_word = sorted_list[0][0]
        self.bki_word = bki_word
        flags = []
        for sus_words_li in self.all_sus_words_li:
            if bki_word in sus_words_li:
                flags.append(1)
            else:
                flags.append(0)
        filter_train = []
        for i in range(len(nls)):
            if flags[i] == 0:
                filter_train.append([nls[i], codes[i]])
        df = pd.DataFrame(filter_train, columns=['src', 'tgt'])
        df.to_csv('BKI_' + task + 'train_poisoned_' + prob + '.csv')

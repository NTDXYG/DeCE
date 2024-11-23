import math
import os
import torch
import torch
from badam import BlockOptimizer
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, AutoModelForCausalLM, \
    get_linear_schedule_with_warmup, T5ForConditionalGeneration, RobertaModel, RobertaConfig, RobertaTokenizer
from custom_datasets import GPTDataset, EncDecDataset
import pandas as pd
import warnings
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from defense_loss import DeCE

warnings.filterwarnings("ignore")

class DecModel():

    def __init__(self, model_path, load_path="None", source_len=256, cutoff_len=512):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cutoff_len = cutoff_len
        self.source_len = source_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if load_path == "None":
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.model.to(self.device)
    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, output_dir,
              loss_func, optimizer):

        train_data = GPTDataset(train_filename, tokenizer=self.tokenizer, source_len=self.source_len,
                                cutoff_len=self.cutoff_len)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        if optimizer == 'badamw':
            optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
            optimizer = BlockOptimizer(
                base_optimizer=optimizer,  # can be any torch.Optimizer
                named_parameters_list=list(self.model.named_parameters()),
                switch_block_every=250,
                # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter.
                switch_mode="random",
                # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
                verbose=2,  # information level, will print trainable parameters when setting to 2
                # include_embedding=True,
                # include_lm_head=True,
            )
        else:
            optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_loss = 0, 1e6
        count = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, labels=labels)
                if loss_func == 'ce':
                    lm_logits = outputs.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_func = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                    loss = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    lm_logits = outputs.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # print('DeCE...')
                    loss_fct = DeCE(label_smoothing=0.05, alpha_base=0.8, ignore_index=self.tokenizer.pad_token_id)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), cur_epoch + 1)

                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.cuda.empty_cache()
    
    def predict(self, source):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.source_len, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            summary_text_ids = self.model.generate(input_ids=source_ids, max_length=self.cutoff_len-self.source_len, do_sample=False)
            t = summary_text_ids[0].cpu().numpy()[:, source_ids.shape[-1]:]
            text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def test(self, test_filename, output_dir):
        df = pd.read_csv(test_filename)
        srcs = df['src'].tolist()
        hyps = []
        for src in srcs:
            hyp = self.predict(src)
            hyps.append(hyp)
        df = pd.DataFrame(hyps, columns=['hyp'])
        df.to_csv(output_dir+'/hyps.csv', index=False)

class EncDecModel():

    def __init__(self, model_path, load_path="None", source_len=256, target_len=256):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_len = target_len
        self.source_len = source_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if load_path == "None":
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(load_path, torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True)

        self.model.to(self.device)

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, output_dir,
              loss_func):

        train_data = EncDecDataset(train_filename, tokenizer=self.tokenizer, source_len=self.source_len,
                                target_len=self.target_len)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_loss = 0, 1e6

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, labels=labels)
                if loss_func == 'ce':
                    lm_logits = outputs.logits
                    loss_func = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                    loss = loss_func(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                else:
                    lm_logits = outputs.logits
                    loss_fct = DeCE(label_smoothing=0.05, alpha_base=0.8, ignore_index=self.tokenizer.pad_token_id)
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), cur_epoch + 1)

                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.cuda.empty_cache()
    
    def predict(self, source):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.source_len, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        source_mask = encode['attention_mask'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            summary_text_ids = self.model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=self.target_len, do_sample=False)
            t = summary_text_ids[0].cpu().numpy()
            text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def test(self, test_filename, output_dir):
        df = pd.read_csv(test_filename)
        srcs = df['src'].tolist()
        hyps = []
        for src in srcs:
            hyp = self.predict(src)
            hyps.append(hyp)
        df = pd.DataFrame(hyps, columns=['hyp'])
        df.to_csv(output_dir+'/hyps.csv', index=False)


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, target_ids=None, loss_func='ce', cur_epoch=0, args=None):
        outputs = self.encoder(source_ids)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        # generate the source_mask and target_mask for tokenizer.pad_token_id
        source_mask = source_ids.eq(self.tokenizer.pad_token_id)
        target_mask = target_ids.eq(self.tokenizer.pad_token_id)

        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            if loss_func == 'ce':
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])
            else:
                loss_fct = DeCE(label_smoothing=0.05, alpha_base=0.8, ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss], cur_epoch + 1)
            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()

                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class EncModel():

    def __init__(self, codebert_path, load_path="None", source_len=256, target_len=256):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_len = target_len
        self.source_len = source_len

        config = RobertaConfig.from_pretrained(codebert_path)
        encoder = RobertaModel.from_pretrained(codebert_path)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        if load_path == "None":
            self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=1, max_length=self.target_len,
                        sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id)
        else:
            self.model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=1, max_length=self.target_len,
                        sos_id=self.tokenizer.cls_token_id, eos_id=self.tokenizer.sep_token_id)
            self.model.load_state_dict(torch.load(load_path))

        self.model.to(self.device)

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, output_dir,
              loss_func):

        train_data = EncDecDataset(train_filename, tokenizer=self.tokenizer, source_len=self.source_len,
                                target_len=self.target_len)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_loss = 0, 1e6

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    _, loss, num = self.model(source_ids=input_ids, target_ids=labels, loss_func=loss_func, cur_epoch=cur_epoch)
                        
                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.cuda.empty_cache()

    def predict(self, source):
        encode = self.tokenizer.encode_plus(source, return_tensors="pt", max_length=self.max_source_length, truncation=True, pad_to_max_length=True)
        source_ids = encode['input_ids'].to(self.device)
        source_mask = encode['attention_mask'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            summary_text_ids = self.model(source_ids=source_ids, source_mask=source_mask)
            t = summary_text_ids[0][0].cpu().numpy()
            text = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def test(self, test_filename, output_dir):
        df = pd.read_csv(test_filename)
        srcs = df['src'].tolist()
        hyps = []
        for src in srcs:
            hyp = self.predict(src)
            hyps.append(hyp)
        df = pd.DataFrame(hyps, columns=['hyp'])
        df.to_csv(output_dir+'/hyps.csv', index=False)
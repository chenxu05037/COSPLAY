import math
import os
import pickle
from collections import deque, defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from code_structure import LOAD_CONCEPT_DIST_MATRIX, CALCULATE_CONCEPT_SET, COSPLAY_PRED, COMMON_GROUND_REWARD, \
    PERSONA_RECALL_SCORE
from parlai.core.agents import Agent, create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.thread_utils import SharedTable
from agents.cosplay.cosplay import ARCH_CHOICE
from agents.cosplay.cosplay import GPTDictionaryAgent, Gpt2SeqModel
from agents.cosplay.utils import maintain_dialog_history, PaddingUtils, round_sigfigs
from agents.cosplay.seq2seq.model import Seq2seqModel
from torch.optim import SGD, lr_scheduler, Adagrad
from agents.utils.utils import split_pad_vector, split_pad_vector_for_bug
from agents.cosplay_rl.utils import prepare_for_understand
from agents.common.dict_helper import SpecialToken
from agents.cosplay.gpt.loss import TokenCrossEntropyLoss
from agents.cosplay.gpt.optim import GPTOptimizer
# import pysnooper
from torch import autograd
from os import path

import json
from agents.cosplay_rl.utils import LanguageModel
from concept_set_framework import prepare_example_persona_kws, prepare_example_for_kw_model, cal_concept2word_map, \
    cal_concept_set, load_concept_dist_matrix, create_concept_dist_matrix, set_union_operation, set_dist_operation, \
    concept_set, extract_concepts, persona_recall_score
from concept_set_framework import prepare_batch_persona_concept_mask, prepare_batch_for_kw_model
from concept_set_framework import cal_word2concept_map, get_keyword_mask_matrix
import torch.nn as nn

task_key_word = 'OriginalPersonaTeacher'


def _setup_op(opt, component):
    assert component in ['cosplay', 'utils']
    new_opt = deepcopy(opt)
    for k, v in opt.items():
        if k.endswith(component):
            new_k = k[:-len(component) - 1]
            new_opt[new_k] = v
    return new_opt


class CosplayRLAgent(Agent):
    @staticmethod
    def dictionary_class():
        if 'gpt' in ARCH_CHOICE:
            return GPTDictionaryAgent
        else:
            return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Specific arguments for this agent"""
        agent = argparser.add_argument_group('DSquare Arguments')
        agent.add_argument('--init-model-cosplay', type=str, default=None,
                           help='Load weights from pre-trained cosplay')

        # cosplay argument
        agent.add_argument('-bzt', '--batchsize-cosplay')
        agent.add_argument('-lrt', '--lr-cosplay')
        agent.add_argument('--gradient-clip', type=float, default=5.0)
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seqModel.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-ehst', '--encoder-hidden-size-cosplay', type=int, default=1024,
                           help='size of the hidden layers')
        agent.add_argument('-dhst', '--decoder-hidden-size-cosplay', type=int, default=1024,
                           help='size of the hidden layers')

        agent.add_argument('-eeszt', '--encoder-embed-dim-cosplay', type=int, default=300,
                           help='size of the token embeddings')
        agent.add_argument('-deszt', '--decoder-embed-dim-cosplay', type=int, default=300,
                           help='size of the token embeddings')

        agent.add_argument('-enlt', '--encoder-layers-cosplay', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dnlt', '--decoder-layers-cosplay', type=int, default=2,
                           help='number of hidden layers')

        agent.add_argument('--rnn-share-cosplay', default=False)

        agent.add_argument('--dropout-cosplay', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--encoder-dropout-in-cosplay', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--encoder-dropout-out-cosplay', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--decoder-dropout-in-cosplay', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--decoder-dropout-out-cosplay', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--share-decoder-input-output-embed-cosplay', type=bool, default=True)

        agent.add_argument('-bit', '--encoder-bidirectional-cosplay', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-attt', '--decoder-attention-cosplay', default='general',
                           choices=['none', 'concat', 'general', 'dot', 'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attlt', '--attention-length-cosplay', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time-cosplay', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rct', '--rank-candidates-cosplay', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the prob score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-rnnt', '--rnn-class-cosplay', default='lstm',
                           choices=Seq2seqModel.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dect', '--decoder-cosplay', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-ltt', '--lookuptable-cosplay', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-embt', '--embedding-type-cosplay', default='random',
                           choices=['random', 'glove', 'glove-fixed',
                                    'fasttext', 'fasttext-fixed',
                                    'glove-twitter'],
                           help='Choose between different strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove or '
                                'Fasttext.'
                                'Preinitialized embeddings can also be fixed '
                                'so they are not updated during training.')
        agent.add_argument('-softt', '--numsoftmax-cosplay', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-optr', '--optimizer-cosplay', default='gpt2_custom')
        agent.add_argument('-beam', '--beam-size-cosplay', default=1, help='cosplay beam size')
        agent.add_argument('--encoder_dis_use', type=bool, default=False,
                           help='add distance state embedding')
        agent.add_argument('--encoder_turn_use', type=bool, default=False,
                           help='add turn state embedding')
        agent.add_argument('--encoder_turn_dim', type=int, default=0,
                           help='encoder turn dimension')

        agent.add_argument('--use-persona-token', type=bool, default=True,
                           help='add special tokens at the start and end of persona')
        agent.add_argument('--use-talk-token', type=bool, default=True,
                           help='add special tokens at the start and end of query')
        agent.add_argument('--persona-append-strategy', default='concat', choices=['concat', 'none', 'select'],
                           help='add special tokens at the start and end of query')
        agent.add_argument('--history-append-strategy', type=int, default=-1,
                           help='-1 means all history are appended, and 0 means none, either')

        agent.add_argument('--share-encoder-persona-dialogue', type=bool, default=True,
                           help='share the same encoder when encoding dialogue and persona')
        agent.add_argument('--encode_max_seq_len', type=int, default=256)
        agent.add_argument('--decode_max_seq_len', type=int, default=24)
        agent.add_argument('--shuffle_persona', type=bool, default=True)

        # Cosplay-rl opt
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=True,
                           help='rank candidates if available. this is done by'
                                ' computing the prob score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=256,
                           help='truncate input & output lengths to speed up '
                                'training (may reduce accuracy). This fixes all '
                                'input and output to have a maximum length. This '
                                'reduces the total amount '
                                'of padding in the batches.')
        agent.add_argument('-pt', '--person-tokens', type='bool', default=False,
                           help='use special tokens before each speaker')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('-gpu', '--gpu', type=int, default=0,
                           help='which GPU device to use')
        agent.add_argument('-blf', '--beam_log_freq', default=0, help='log beam search results')

        agent.add_argument('--gradient_clip', type=int, default=1.0)
        agent.add_argument('--batchsize', type=int, default=1)
        agent.add_argument('--top-k', type=int, default=1,
                           help='sample number for random topk word in seq2seq generation')
        # report setting
        agent.add_argument('-rmetric', '--train-report-metrics', default='',
                           help='which metrics you need to see when training DSquare')

        # dict setting
        """ Add option for dictionary agent"""
        agent.add_argument('--dict-nulltoken', default=SpecialToken.pad)
        agent.add_argument('--dict-starttoken', default=SpecialToken.start)
        agent.add_argument('--dict-endtoken', default=SpecialToken.end)
        agent.add_argument('--dict-unktoken', default=SpecialToken.unk)
        agent.add_argument('--dict-tokenizer', default='split')
        agent.add_argument('--dict-language', default='english')
        agent.add_argument('--dict-include-valid', type=bool, default=True)
        agent.add_argument('--dict_file', default='../../tmp/dict/convai2_self_seq2seq_model.dict')
        agent.add_argument('--dict_lower', type=bool, default=True)

        CosplayRLAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared=None)
        opt = self.opt

        self.batch_size = opt['batchsize']
        self.history = [{} for _ in range(self.batch_size)]  # dialogue history for act
        self.log_probs = []
        self.is_training = None
        self.is_first_speaker = None

        # if greedy response, one is teacher as validation, another is sampling.
        self.greedy_response = None

        self.persona_receiver = None  # text list which describes the agent's persona
        self.persona_cosplay = None  # text list which is used by cosplay
        self.send_messages, self.receive_messages = [], []

        self.encode_max_seq_len = opt['encode_max_seq_len'] if opt['encode_max_seq_len'] > 0 else None
        self.decode_max_seq_len = opt['decode_max_seq_len'] if opt['decode_max_seq_len'] > 0 else None

        # batch share the same persona information
        self.use_person_tokens = opt.get('use_persona_token', True)
        self.use_talk_tokens = opt.get('use_talk_token', True)
        self.use_history_reply = opt.get('history_replies', 'label_else_model')
        self.add_default_persona = opt.get('add_default_persona', False)
        self.persona_append_strategy = opt.get('persona_append_strategy', 'concat')
        self.history_append_strategy = opt.get('history_append_strategy', -1)

        self.metrics = {'loss': 0.0,
                        'reward': 0.0,
                        'reward_var': 0.0,
                        'total_reward': 0.0,
                        'correct_tokens': 0,
                        'num_tokens': 0,
                        'num_selfplay_turns': 0,
                        'num_selfplay_episode': 0,
                        'total_skipped_batches': 0}
        if opt['train_report_metrics'] is None or opt['train_report_metrics'] == '':
            self.report_metrics = self.metrics.keys()
        else:
            self.report_metrics = set(opt['train_report_metrics'].split(','))

        self.rank = opt['rank_candidates']
        self.topk = opt.get('top_k', 1)
        self.report_freq = opt.get('report_freq', 0.001)
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()

        if shared:  # TODO:
            # set up shared properties
            self.opt = shared['opt']
            opt = self.opt
            self.dict = shared['dict']

            # idea interface
            self.device = shared['device']
            self.word2concept_map = shared['word2concept_map']
            self.concept2words_map = shared['concept2words_map']
            self.kw_mask_matrix = shared['kw_mask_matrix']
            self.kw_graph_distance_matrix = shared['kw_graph_distance_matrix']

            self.START_IDX = shared['START_IDX']
            self.END_IDX = shared['END_IDX']
            self.NULL_IDX = shared['NULL_IDX']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']

            # if 'cosplay' in shared:
            # model is shared during hogwild
            self.cosplay = shared['cosplay']
            self.metrics = shared['metrics']
            # TODO: share status

            # language model to score
            self.coherent_model = shared['coherent_model']
            self.language_model = shared['language_model']
            self.id = 'Cosplay Agent'
        else:
            if self.use_cuda:
                print('[ Using CUDA (GPU:{})]'.format(opt['gpu']))
                torch.cuda.set_device(opt['gpu'])
                self.device = torch.device('cuda')
            else:
                print('[ Using CPU ]')
                self.device = torch.device('cpu')

            self.answers = [None] * self.batch_size
            self.id = 'Cosplay Agent'
            self.dict = self.dictionary_class()(opt)

            # idea interface
            self.word2concept_map = cal_word2concept_map(self.dict, self.device)
            self.concept2words_map = cal_concept2word_map(self.word2concept_map, self.device)

            self.kw_mask_matrix = get_keyword_mask_matrix(self.device)

            LOAD_CONCEPT_DIST_MATRIX()  # 1. load matrix if the pkl file exists
            concept_dist_matrix_pkl = '{}/concept_net/concept_dist_matrix.pkl'.format(self.opt['datapath'])
            if path.exists(concept_dist_matrix_pkl):
                self.kw_graph_distance_matrix = load_concept_dist_matrix(concept_dist_matrix_pkl)
            else:
                self.kw_graph_distance_matrix = create_concept_dist_matrix(
                    self.opt['datapath'] + '/concept_net/keyword_graph_weighted_distance_dict.pkl', self.device)

            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            cosplay_status, receiver_status, language_status = {}, {}, {}

            # model initialization
            init_cosplay, init_receiver = None, None
            if opt.get('init_model_cosplay') and os.path.isfile(opt['init_model_cosplay']):
                init_cosplay = opt['init_model_cosplay']

            if init_cosplay is not None:
                print('[ Loading cosplay from {} ]'.format(init_cosplay))
                cosplay_status = self.load(init_cosplay)

            # cosplay_opt = _setup_op(opt, component='cosplay')

            if ARCH_CHOICE == 'lstm':
                self.cosplay = Seq2seqModel(opt=self.opt,
                                            num_features=len(self.dict),
                                            padding_idx=self.NULL_IDX,
                                            start_idx=self.START_IDX,
                                            end_idx=self.END_IDX,
                                            longest_label=cosplay_status.get('longest_label', 1))
            elif ARCH_CHOICE == 'gpt':
                assert isinstance(self.dict, GPTDictionaryAgent)
                self.cosplay = Gpt2SeqModel(opt=self.opt,
                                            vocab_size=len(self.dict),
                                            pad_idx=self.NULL_IDX,
                                            start_idx=self.START_IDX,
                                            end_idx=self.END_IDX,
                                            dict=self.dict,
                                            special_token_len=len(self.dict.special_tokens),
                                            longest_label=cosplay_status.get('longest_label', 1),
                                            device=self.device)

            if opt.get('init_model_coherent') and os.path.isfile(opt['init_model_coherent']):
                init_language = opt['init_model_coherent']
            else:
                init_language = None

            if init_language is not None:
                print('[ Loading coherent model from {} ]'.format(init_language))
                language_status, language_opt = self.load(init_language, override=False)
                language_opt['gpu'] = 0

            coherent_gpt_model = Gpt2SeqModel(opt=language_opt,
                                              vocab_size=len(self.dict),
                                              pad_idx=self.NULL_IDX,
                                              start_idx=self.START_IDX,
                                              end_idx=self.END_IDX,
                                              dict=self.dict,
                                              special_token_len=len(self.dict.special_tokens),
                                              longest_label=80,
                                              device=self.device)

            self.coherent_model = coherent_gpt_model

            if language_status:
                self.coherent_model.load_state_dict(language_status['model'])
                self.coherent_model.eval()

            self.language_model = LanguageModel(pad_idx=self.NULL_IDX)

            if cosplay_status:
                self.cosplay.load_state_dict(cosplay_status['model'])

            if self.use_cuda:
                print('[ Using CUDA ]')
                self.cosplay.cuda()
                self.coherent_model.cuda()
                self.language_model.cuda(self.device)
                self.device = torch.device('cuda')
            else:
                print('[ Using CPU ]')
                self.device = torch.device('cpu')

        self.shuffle_persona = opt['shuffle_persona']
        self.rank = opt['rank_candidates']
        # used in train and validation
        self.criterion = TokenCrossEntropyLoss(pad_index=self.NULL_IDX)

        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        if 'lstm' in ARCH_CHOICE:
            self.cosplay_optimizer = Adagrad(self.cosplay.parameters(),
                                             opt.get('lr', 1e-3))  # TODO: other optimizer
        elif 'gpt' in ARCH_CHOICE:
            self.cosplay_optimizer = GPTOptimizer(self.cosplay, self.opt)
        else:
            raise Exception("Not support for ARCH")

        self.super_optimizer = SGD(params=self.cosplay.parameters(),
                                   lr=opt.get('lr', 1e-2))

        if not isinstance(self.cosplay_optimizer, GPTOptimizer):
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.cosplay_optimizer, 'min', factor=0.5, patience=3, verbose=True)  # TODO: schedule

        self.reset()
        # zero gradient for update
        self.cosplay_optimizer.zero_grad()
        self.super_optimizer.zero_grad()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'batchsize', 'model', 'model_file', 'evaltask', 'eval_batchsize', 'validation_metric',
                      'validation_metric_mode', 'top_k', 'beam_size', 'gpt_lr',
                      'tensorboard_metrics', 'validation_patience',
                      'rank_candidates', 'log_every_n_secs',
                      'dict_maxexs', 'decode_max_seq_len',
                      'encode_max_seq_len', 'lr', 'dict_lower', 'exp', 'train_report_metrics', 'starttime',
                      'gradient_clip', 'init_model_cosplay ', 'optimizer_step'}
        for k, v in new_opt.items():
            # override ones
            if k in model_args and k in self.opt:
                # skip non-model args
                continue
            if k not in self.opt:
                print('[ Adding new option: | {k}: {v} | ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('[ Overriding option: | {k}: {old} => {v} | ]'.format(
                    k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        if 'dict_file' in new_opt and not self.opt.get('dict_file'):
            print('[ No dictionary path detected, trying to load previous '
                  'path {} ]'.format(new_opt['dict_file']))
            self.opt['dict_file'] = new_opt['dict_file']
        return self.opt

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()

        # idea interface
        shared['device'] = self.device
        shared['word2concept_map'] = self.word2concept_map
        shared['concept2words_map'] = self.concept2words_map
        shared['kw_mask_matrix'] = self.kw_mask_matrix
        shared['kw_graph_distance_matrix'] = self.kw_graph_distance_matrix

        shared['opt'] = self.opt
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        # if self.opt.get('numthreads', 1) > 1:
        # we're doing hogwild so share the model too
        if type(self.metrics) == dict:
            # move metrics and model to shared memory
            self.metrics = SharedTable(self.metrics)
            self.cosplay.share_memory()
        shared['cosplay'] = self.cosplay
        shared['metrics'] = self.metrics
        shared['states'] = {  # don't share optimizer states
            'optimizer_type': self.opt.get('optimizer', None),
        }

        shared['coherent_model'] = self.coherent_model
        shared['language_model'] = self.language_model
        return shared

    def set_id(self, new_id=None, suffix=None):
        if new_id is not None:
            self.id = new_id
        if suffix is not None:
            self.id += suffix

    def set_mode(self, is_training):
        self.is_training = is_training

    def set_greedy(self):
        self.greedy_response = True

    def reset(self):
        """Necessary initialization at the beginning of each episode"""
        # TODO: Ensure that world call this function at the beginning of each episode
        self.observation = None
        self.is_training = None
        self.is_first_speaker = None
        self.greedy_response = None
        self.persona_receiver = None
        self.persona_cosplay = None
        self.history = [{} for _ in range(self.batch_size)]
        self.log_probs = []
        self.send_messages = []
        self.receive_messages = []
        for i in range(len(self.answers)):
            self.answers[i] = None
        # self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        self.metrics['loss'] = 0.0  # loss metric is only used to calculated perplexity during evaluation
        self.metrics['reward'] = 0.0
        self.metrics['reward_var'] = 0.0
        self.metrics['total_reward'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self.metrics['num_selfplay_turns'] = 0
        self.metrics['num_selfplay_episode'] = 0
        self.metrics['total_skipped_batches'] = 0

    def report(self):
        """Report training details"""
        if self.is_training:
            ret_metrics = {metric_key: self.metrics[metric_key] for metric_key in self.report_metrics}
            if 'reward_var' in ret_metrics and ret_metrics['num_selfplay_episode'] != 0:
                ret_metrics['reward_var'] = ret_metrics['reward_var'] / ret_metrics['num_selfplay_episode']
            if 'total_reward' in ret_metrics and ret_metrics['num_selfplay_episode'] != 0:
                ret_metrics['total_reward'] = ret_metrics['total_reward'] / ret_metrics['num_selfplay_episode']
            return ret_metrics
        else:
            m = {}
            num_tok = self.metrics['num_tokens']
            if num_tok > 0:
                if self.metrics['correct_tokens'] > 0:
                    m['token_acc'] = self.metrics['correct_tokens'] / num_tok
                m['loss'] = self.metrics['loss'] / num_tok
                try:
                    m['ppl'] = math.exp(m['loss'])
                except OverflowError:
                    m['ppl'] = float('inf')
            if self.metrics['total_skipped_batches'] > 0:
                m['total_skipped_batches'] = self.metrics['total_skipped_batches']
            for k, v in m.items():
                # clean up: rounds to sigfigs and converts tensors to floats
                m[k] = round_sigfigs(v, 4)
            return m

    def act(self, is_display=False):
        if isinstance(self.observation, dict):
            act = self.batch_act([self.observation])[0]
        else:
            # batch is in the training.
            act = self.batch_act(self.observation)

            # idea interface
            # self.history[labels]

            if 'Cosplay' in act[0]['id'] and 'init' not in self.observation[0][
                'id']:  # Only add dialogue text excluding the persona text
                if len(self.send_messages) == 0:  # At the beginning of an episode
                    if len(self.receive_messages) == 0:  # The first speaker receive no text before it first speaks
                        self.is_first_speaker = True
                    else:
                        self.is_first_speaker = False
                self.send_messages.append([a['text'] for a in act])
            if self.is_training and is_display:  # TODO: clean up logging
                print('[ {} speaks ] {}'.format(self.id, act[0]['text']))

            # idea interface
            for idx, x in enumerate(act):
                self.history[idx]['labels'] = [act[idx]['text']]

        return act

    def batch_act(self, observations):
        # DEBUG track gpu
        reply = [{'id': self.getID(), 'episode_done': False} for _ in range(self.batch_size)]
        src_seq, src_seq_turn, src_seq_dis, tgt_seq, tgt_seq_turn, labels, valid_inds, cands, valid_cands, is_training = self.cosplay_vectorize(
            observations)

        # idea interface
        persona_set = prepare_batch_persona_concept_mask(observations, device=self.device)
        data_for_kw_model = prepare_batch_for_kw_model(observations, device=self.device)
        context_concepts = data_for_kw_model['batch_context_keywords']

        CALCULATE_CONCEPT_SET()
        concept_set = cal_concept_set(self.opt, context_concepts, persona_set, self.kw_graph_distance_matrix,
                                      self.device, self.concept2words_map, k=self.opt.get('persona_pool_size'))

        cand_inds = [i[0] for i in valid_cands] if valid_cands is not None else None

        use_attention = self.opt.get('use_attention')

        COSPLAY_PRED()
        predictions, cand_preds = self.cosplay_predict(src_seq, src_seq_turn, src_seq_dis, tgt_seq, tgt_seq_turn,
                                                       cands, cand_inds, is_training,
                                                       data_for_kw_model=data_for_kw_model,
                                                       persona_kw_mask=persona_set,
                                                       final_pool=concept_set,
                                                       use_attention=use_attention)

        if self.is_training:
            report_freq = 0
        else:
            report_freq = self.report_freq

        if predictions is not None:
            PaddingUtils.map_predictions(
                predictions, valid_inds, reply, observations,
                self.dict, self.END_IDX, report_freq=report_freq, labels=labels,
                answers=self.answers, ys=tgt_seq.data if tgt_seq is not None else None)

        # TODO: check
        # self.answers = [self.dict.vec2txt(ans) for ans in self.answers] # convert back to txt for history tracking

        if cand_preds is not None:
            if valid_cands is None:
                valid_cands = [(None, i, labels) for i in valid_inds]
            for i in range(len(valid_cands)):
                order = cand_preds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]
        return reply

    def observe(self, observation):
        """Save observation for act"""
        obs = observation.copy()

        if self.is_training:
            assert type(observation) == list
        else:
            assert type(observation) == dict
            obs = [obs]

        first_obs = obs[0]

        if not first_obs.get('preprocessed', False) or 'text2vec' not in first_obs:
            # preprocess text in observations, replace candidates persona into another tagging
            for ob_ind, ob in enumerate(obs):
                if ob.get('text', False):
                    # TODO: not consistent with cosplay
                    text_split = ob['text'].lower().split('\n')
                    persona_given = ''
                    for t in text_split:
                        if t.startswith('your persona') or t.startswith('partner\'s persona'):
                            t = t.replace('your persona: ', '').replace('partner\'s persona: ', '')
                            persona_given += t + '\n'
                    # TODO: drop the old method to encode persona, we use the separate encoder to do that.

                    # idea interface
                    # if self.answers[0] is not None:
                    #     if self.use_talk_tokens:
                    #         # idea modified ob['text'] = SpecialToken.talk_1_start + ' ' + text_split[-1] + ' ' + SpecialToken.talk_1_end
                    #         ob['text'] = SpecialToken.talk_1_start + ' ' + self.answers[
                    #             ob_ind] + ' ' + SpecialToken.talk_1_end
                    #     else:
                    #         ob['text'] = self.answers[ob_ind]
                    # else:
                    if text_split[-1].startswith('your persona') or text_split[-1].startswith('partner\'s persona'):
                        ob['text'] = None
                    else:
                        if self.use_talk_tokens:
                            ob['text'] = SpecialToken.talk_1_start + ' ' + text_split[
                                -1] + ' ' + SpecialToken.talk_1_end
                        else:
                            ob['text'] = self.answers[ob_ind]

                    if persona_given != '':
                        # self.persona_receiver = ' '.join([' ' + text.strip() + ' ' + self.receiver_dict.end_token
                        #                                   for text in persona_given.split('\n')
                        #                                   if text != ''])
                        split_persona = persona_given.split('\n')
                        persona_text = ' '.join(split_persona)
                        if self.use_person_tokens:
                            persona_text = SpecialToken.persona_start + ' ' + persona_text + ' ' + SpecialToken.persona_end
                        self.persona_cosplay = persona_text
                    ob['persona'] = persona_given
                    # idea interface
                    ob['persona_kws'] = prepare_example_persona_kws(self.history[ob_ind], persona_given)

                    if self.send_messages:
                        message = self.send_messages[0][ob_ind]
                    else:
                        message = None

                    ob['kw_model'] = prepare_example_for_kw_model(message, text_split[-1], self.dict)

            for ob_ind, ob in enumerate(obs):
                ob['text2vec'], ob['dis2vec'], ob['turn2vec'], ob['cur_turn'] = maintain_dialog_history(
                    self.history[ob_ind], ob,
                    reply=self.answers[ob_ind],
                    max_history_len=self.encode_max_seq_len,
                    use_reply=self.use_history_reply,
                    persona_append_strategy=self.persona_append_strategy,
                    history_append_strategy=self.history_append_strategy,
                    use_persona_tokens=self.use_person_tokens,
                    shuffle_persona=self.is_training and self.shuffle_persona,
                    dict=self.dict)
        else:
            for ob in obs:
                ob['text2vec'] = deque(ob['text2vec'], maxlen=self.truncate)
                ob['turn2vec'] = deque(ob['turn2vec'], maxlen=self.truncate)

        if self.is_training and 'Cosplay' in first_obs['id'] and not first_obs['episode_done']:
            # observe text from the other interlocutor
            self.receive_messages.append([ob['text'] for ob in obs])
            # obs['text'] = self.persona + obs['text'] # Adding persona to the beginning of each turn

        if self.is_training:
            self.observation = obs
        else:
            # restore the dictionary
            self.observation = obs[0]

        if self.is_training and first_obs['episode_done'] \
                and 'reward' in first_obs and 'Cosplay' in first_obs['id'] and not self.greedy_response:
            # end of an self-play episode, backwards propagation
            # assume each turn, each word has similar proportion of rewards
            rewards = np.stack([o['reward'] for o in obs])  # average the reward over turns
            # batch_size x turn_size
            log_probs = torch.cat(self.log_probs).transpose(0, 1).contiguous()
            try:
                with autograd.detect_anomaly():
                    # concat
                    reward_tensor = torch.tensor(rewards, device=log_probs.device, requires_grad=False).float().view(-1)
                    scores = (-log_probs.view(-1)) * reward_tensor
                    loss = scores.mean()
                    loss.backward()
                    self.metrics['reward'] = loss.item()
                    self.metrics['reward_var'] += float(np.std(rewards, axis=0).mean())
                    self.metrics['total_reward'] += self.metrics['reward']
                    self.metrics['num_selfplay_episode'] += 1
            except RuntimeError as e:
                print("Expection: {}".format(e))
                print("[ERROR in NAN problem. Not update this time.]")

        return self.observation

    def cosplay_predict(self, src_seq, src_seq_turn, src_seq_dis, tgt_seq=None, tgt_seq_turn=None, cands=None,
                        valid_cands=None, is_training=False, data_for_kw_model=None, persona_kw_mask=None,
                        final_pool=None, use_attention=None, visualization=False):
        """Produce a prediction from our cosplay.

        Keep track of gradients if is_training
        """
        # gpu_tracker.track()
        predictions, cand_preds = None, None

        # idea interface: for both train and generation codes.
        # kw_logits, kw_hidden_states = cal_kw_logits(data_for_kw_model, self.kw_mask_matrix, self.cosplay.kw_model)
        # walk_probs = cal_next_pool(kw_logits, self.kw_mask_matrix,
        #                            data_for_kw_model['batch_context_keywords'], self.cosplay.softmax)
        # jump_probs = cal_persona_pool(self.kw_graph_distance_matrix, persona_kw_mask, self.cosplay.softmax)
        # hybrid_weights = self.opt['hybrid_weights']

        if is_training:
            self.cosplay.train()
            try:
                if tgt_seq is not None:
                    out = self.cosplay.forward(src_seq=src_seq,
                                               src_seq_turn=src_seq_turn,
                                               src_seq_dis=src_seq_dis,
                                               tgt_seq=tgt_seq,
                                               tgt_seq_turn=tgt_seq_turn,
                                               word2concept_map=self.word2concept_map,
                                               concept2words_map=self.concept2words_map,
                                               final_pool=final_pool,
                                               visualization=visualization,
                                               use_attention=use_attention)
                    predictions, hybrid_probs, cand_preds, gate = out[0], out[1], out[2], out[4]
                    # idx = predictions.unsqueeze(dim=2)
                    # loss = self.criterion(scores, idx)

                    # calculate gradient
                    y_ne = tgt_seq.ne(self.NULL_IDX)
                    target_tokens = y_ne.long().sum().item()
                    correct = ((tgt_seq == predictions) * y_ne).sum().item()
                    self.metrics['num_tokens'] += target_tokens
                    self.metrics['correct_tokens'] += correct

                    gen_loss_fn = nn.NLLLoss(ignore_index=0, reduction='mean')
                    gen_loss = gen_loss_fn(hybrid_probs.log().view(-1, hybrid_probs.size(-1)), tgt_seq.view(-1))

                    # loss /= target_tokens  # average loss per token
                    gen_loss.backward()
                elif self.greedy_response is True:  # 2pegg B
                    out = self.cosplay.forward(src_seq=src_seq,
                                               src_seq_turn=src_seq_turn,
                                               src_seq_dis=src_seq_dis,
                                               sampling=False,
                                               word2concept_map=self.word2concept_map,
                                               concept2words_map=self.concept2words_map,
                                               final_pool=final_pool,
                                               use_attention=use_attention)
                    # generated response
                    predictions, hybrid_probs, cand_preds = out[0], out[1], out[2]
                    self.metrics['num_selfplay_turns'] += 1
                else:  # 2pegg for A sampling
                    out = self.cosplay.forward(src_seq=src_seq,
                                               src_seq_turn=src_seq_turn,
                                               src_seq_dis=src_seq_dis,
                                               sampling=True,
                                               word2concept_map=self.word2concept_map,
                                               concept2words_map=self.concept2words_map,
                                               final_pool=final_pool,
                                               use_attention=use_attention)
                    # generated response
                    predictions, hybrid_probs, cand_preds = out[0], out[1], out[2]
                    idx = predictions.unsqueeze(dim=2)
                    # keep the same with PyTorch bernoulli distribution, here take the log probability
                    lprobs = hybrid_probs.log()
                    # print("lprobs: {}, shape: {}".format(lprobs, lprobs.size()))
                    zero_mask = predictions.ne(self.NULL_IDX).float()
                    log_prob = torch.gather(lprobs, 2, idx).squeeze(2)
                    log_prob = (log_prob * zero_mask).sum(1)
                    # avoid dividing zero !
                    actual_len = predictions.ne(self.NULL_IDX).float().sum(1)
                    log_prob /= actual_len  # TODO: length penalty

                    # scores: Tensor of shape (1, len_sentence, size_vocabualry)
                    self.log_probs.append(log_prob.unsqueeze(dim=0))
                    self.metrics['num_selfplay_turns'] += 1

            except RuntimeError as e:
                # catch out of memory exceptions during fwd/bck (skip batch)
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch. '
                          'if this happens frequently, decrease batchsize or '
                          'truncate the inputs to the model.')
                    self.metrics['total_skipped_batches'] += 1
                    return predictions, cand_preds
                else:
                    raise e
        else:
            with torch.no_grad():
                self.cosplay.eval()
                out = self.cosplay.forward(src_seq=src_seq,
                                           src_seq_turn=src_seq_turn,
                                           src_seq_dis=src_seq_dis,
                                           rank_during_training=cands is not None,
                                           cands=cands,
                                           valid_cands=valid_cands,
                                           word2concept_map=self.word2concept_map,
                                           concept2words_map=self.concept2words_map,
                                           final_pool=final_pool,
                                           use_attention=use_attention)
                predictions, cand_preds = out[0], out[2]
                if tgt_seq is not None:
                    # calculate loss on targets
                    out = self.cosplay.forward(src_seq=src_seq,
                                               src_seq_turn=src_seq_turn,
                                               src_seq_dis=src_seq_dis,
                                               tgt_seq=tgt_seq,
                                               tgt_seq_turn=tgt_seq_turn,
                                               cands=cands,
                                               valid_cands=valid_cands,
                                               word2concept_map=self.word2concept_map,
                                               concept2words_map=self.concept2words_map,
                                               final_pool=final_pool,
                                               use_attention=use_attention)
                    # scores = out[1]
                    # loss = self.criterion(scores, tgt_seq)
                    hybrid_probs = out[1]
                    gen_loss_fn = nn.NLLLoss(ignore_index=0, reduction='sum')
                    gen_loss = gen_loss_fn(hybrid_probs.log().view(-1, hybrid_probs.size(-1)), tgt_seq.view(-1))
                    # save loss to metrics
                    target_tokens = tgt_seq.ne(self.NULL_IDX).long().sum().item()
                    self.metrics['loss'] += gen_loss.item()
                    self.metrics['num_tokens'] += target_tokens
        # gpu_tracker.track()
        torch.cuda.empty_cache()
        return predictions, cand_preds

    def cosplay_vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = self.is_training

        src_seq, tag_seq, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=self.END_IDX,
            null_idx=self.NULL_IDX, dq=True, eval_labels=True,
            encode_truncate=self.encode_max_seq_len, decode_truncate=self.decode_max_seq_len)

        max_seq_len = len(src_seq[0])
        # now the source sequence turn is just `relative distance`
        src_seq_dis = []
        # TODO: add turn embedding for src_seq
        for cur_ind, org_ind in enumerate(valid_inds):
            org_dis_ids = observations[org_ind]['dis2vec'].copy()
            org_dis_ids.extend([0] * (max_seq_len - len(org_dis_ids)))
            src_seq_dis.append(org_dis_ids)

        src_seq_turn = []
        tgt_seq_turn = []
        # TODO: add turn embedding for src_seq
        for cur_ind, org_ind in enumerate(valid_inds):
            org_turn_ids = observations[org_ind]['turn2vec'].copy()
            org_turn_ids.extend([0] * (max_seq_len - len(org_turn_ids)))
            src_seq_turn.append(org_turn_ids)
            # decode turn id as input
            tgt_seq_turn.append(observations[org_ind]['cur_turn'])

        if src_seq is None:
            return None, None, None, None, None, None, None

        src_seq = torch.LongTensor(src_seq)

        if tag_seq is not None:
            tag_seq = torch.LongTensor(tag_seq)

        if self.use_cuda:
            # copy to gpu
            src_seq = src_seq.cuda()
            if tag_seq is not None:
                tag_seq = tag_seq.cuda()

        cands = None
        valid_cands = None

        # rank candidates in validation
        if not is_training and self.rank:
            # set up candidates
            cands = []
            valid_cands = []
            for i, v in enumerate(valid_inds):
                if 'label_candidates' in observations[v]:
                    curr_lcs = list(observations[v]['label_candidates'])
                    curr_cands = [{'text': c} for c in curr_lcs]
                    # padding candidates
                    cs, _, _, valid_c_inds, *_ = PaddingUtils.pad_text(curr_cands, self.dict, null_idx=self.NULL_IDX,
                                                                       # TODO: whether add end idx to add
                                                                       dq=True, encode_truncate=self.decode_max_seq_len)
                    valid_cands.append((i, v, [curr_lcs[j] for j in valid_c_inds]))
                    cs = torch.LongTensor(cs)
                    if self.use_cuda:
                        cs = cs.cuda()
                    cands.append(cs)

        return src_seq, src_seq_turn, src_seq_dis, tag_seq, tgt_seq_turn, labels, valid_inds, cands, valid_cands, is_training

    def update_selfplay(self):
        """Update component given self-play agents' experience(gradient)"""
        # print("Before Clear Self Gradient: {}".format(
        # str(self.cosplay.transformer_module.lm_head.decoder.weight.grad)))
        self.cosplay_optimizer.step()
        self.cosplay_optimizer.zero_grad()

    def update_supervised(self):
        """Update component given self-play agents' experience(gradient)"""
        # print("Before Clear Self Gradient: {}".format(
        # str(self.cosplay.transformer_module.lm_head.decoder.weight.grad)))
        self.super_optimizer.step()
        self.super_optimizer.zero_grad()

    def understand(self):
        """Generate the received dialog_all from the dialogue text"""
        send_messages = deepcopy(self.send_messages)
        receive_messages = deepcopy(self.receive_messages)

        # remove the <end> from message for later robust splitting
        receive_messages = [[message.replace(self.dict.end_token, '') for message in interaction]
                            for interaction in receive_messages]
        send_messages = [[message.replace(self.dict.end_token, '') for message in interaction]
                         for interaction in send_messages]

        is_first_speaker = self.is_first_speaker

        with torch.no_grad():
            dialogue, valid_mask = prepare_for_understand(send_messages, receive_messages, is_first_speaker,
                                                          end_sep=self.receiver_dict.end_token)
            dialog_all, dialog_val = self.receiver.encode_dialog(dialogue, valid_mask=valid_mask,
                                                                 dict_agent=self.receiver_dict, use_cuda=self.use_cuda)
            # use dialog valid to represent send messages
        return dialog_val

    def measure_persona_similarity(self, vt_persona, persona):
        score = self.receiver.measure_persona_similarity(vt_persona, persona)
        return score

    def confess(self):
        """the agent confesses according to its own persona
        :return Tensor, persona embedding
        """
        assert self.persona_receiver is not None
        encoded_persona = self.receiver.encode_persona([self.persona_receiver], self.receiver_dict, self.use_cuda)
        return encoded_persona

    def coherent_score(self, first_message):
        """
        Eval the language model score gained by agent
        :param first_message: list of first messages, should be [turn_size]
        :return:
        """
        receive_messages_list = deepcopy(self.receive_messages)
        send_messages_list = deepcopy(self.send_messages)

        # remove the <end> from message for later robust splitting
        receive_messages_list = [[message.replace(self.dict.end_token, '') for message in interaction]
                                 for interaction in receive_messages_list]
        send_messages_list = [[message.replace(self.dict.end_token, '') for message in interaction]
                              for interaction in send_messages_list]

        if self.is_first_speaker:
            first_message = [SpecialToken.talk_1_start + ' ' + meesage.strip() + ' ' + SpecialToken.talk_1_end for
                             meesage in first_message]
            receive_messages_list.insert(0, first_message)
            receive_messages_list.pop(-1)

        batch_obs = [{'text': '', 'labels': ['']} for _ in range(len(send_messages_list[0]))]
        # prepend self persona into the dialogue history
        batch_history = [self.persona_cosplay for _ in range(len(send_messages_list[0]))]
        for receive_messages, send_messages in zip(receive_messages_list, send_messages_list):
            # for every message
            indexes = range(len(receive_messages))
            for ind, receive_message, send_message in zip(indexes, receive_messages, send_messages):
                if receive_message.strip() == '':
                    # append something
                    receive_message = 'hi'
                if send_message.strip() == '':
                    send_message = 'hi'

                batch_history[ind] += ' ' + receive_message
                batch_obs[ind]['text'] += batch_history[ind] + ' ' + self.dict.end_token + ' '
                batch_obs[ind]['labels'][
                    0] += send_message + ' ' + self.dict.end_token + ' ' + self.dict.start_token + ' '
                batch_history[ind] += ' ' + send_message

        # split and padding into vector
        receive_tensor, send_tensor, _, sort_ind, *_ = PaddingUtils.pad_text(batch_obs, self.dict,
                                                                             null_idx=self.dict.pad_idx,
                                                                             dq=False, eval_labels=True)

        debug_1 = send_tensor

        # batch_size x turn_size x max_seq_len
        receive_tensor = split_pad_vector(receive_tensor, self.dict.end_idx, self.dict.pad_idx)
        receive_tensor = torch.LongTensor(receive_tensor)
        # batch_size x turn_size x max_seq_len
        send_tensor = split_pad_vector_for_bug(send_tensor, self.dict.start_idx, self.dict.pad_idx)
        send_tensor = torch.LongTensor(send_tensor)
        cuda_device = next(self.coherent_model.parameters()).device

        if self.use_cuda:
            receive_tensor = receive_tensor.cuda(cuda_device)
            send_tensor = send_tensor.cuda(cuda_device)

        if receive_tensor.size(1) != send_tensor.size(1):
            print('< debug 1 >')
            [print(i) for i in debug_1]

            print('< debug 2 >')
            print('【receive_tensor size】{}'.format(receive_tensor.size()))
            print('【send_tensor size】{}'.format(send_tensor.size()))
            print('【send_tensor】\n{}'.format(send_tensor))
            send_tensor = send_tensor.view(send_tensor.size(0) * send_tensor.size(1), -1)[
                send_tensor.view(send_tensor.size(0) * send_tensor.size(1), -1).sum(-1).ne(0) * send_tensor.view(
                    send_tensor.size(0) * send_tensor.size(1), -1).sum(-1).ne(40484)].view(send_tensor.size(0), 3, -1)

            print('【send_tensor after】\n{}'.format(send_tensor))

        sorted_score = self.coherent_model.score_sentence(receive_tensor, send_tensor)
        # desorted ind
        desorted_ind = np.array(sort_ind).argsort()
        scores = sorted_score[desorted_ind]
        # baseline
        scores = scores.data.cpu().numpy() - 0.5
        return scores

    def language_score(self):
        """
        Eval the language model score gained by agent
        :return:
        """
        send_messages = deepcopy(self.send_messages)
        send_messages = [[message.replace(self.dict.end_token, '') for message in interaction]
                         for interaction in send_messages]

        batch_messages = ['' for _ in range(len(send_messages[0]))]
        for send_message in send_messages:
            # for every message
            for ind, message in enumerate(send_message):
                batch_messages[ind] += message + ' ' + self.dict.end_token + ' '
        # split and padding into vector
        obs = [{'text': c} for c in batch_messages]
        xs, _, _, sort_ind, *_ = PaddingUtils.pad_text(obs, self.dict,
                                                       null_idx=self.dict.pad_idx,
                                                       dq=False, eval_labels=True)

        debug_xs = xs

        xs = split_pad_vector_for_bug(xs, self.dict.end_idx, self.dict.pad_idx)
        xs = torch.LongTensor(xs)
        cuda_device = next(self.language_model.transformer_module.parameters()).device
        if self.use_cuda:
            xs = xs.cuda(cuda_device)

        # print('【normal shape】 {}'.format(xs.size()))
        assert 3 == xs.size(1)
        try:
            sorted_score = self.language_model.score_sentence(xs)
        except RuntimeError as e:
            print('【debug-1】')
            [print(i) for i in debug_xs]
            print('【debug-2】')
            [print(i) for i in xs.view(-1, xs.size(2)).tolist()]
            print('input: {}'.format(xs))
            print(e)
            print('shape: {}'.format(xs.size()))
            print('input: {}'.format(xs))
            print('input: {}'.format(xs[0]))

        # desorted ind
        desorted_ind = np.array(sort_ind).argsort()
        scores = sorted_score[desorted_ind]
        scores = scores.data.cpu().numpy()
        scores = np.nan_to_num(scores)
        return scores

    def common_ground_reward_and_recall_scores(self, partner_persona, r=None):
        receive_messages_list = deepcopy(self.receive_messages)
        send_messages_list = deepcopy(self.send_messages)

        ## remove the <end> from message for later robust splitting
        receive_messages_list = [[message.replace(self.dict.end_token, '') for message in interaction]
                                 for interaction in receive_messages_list]
        send_messages_list = [[message.replace(self.dict.end_token, '') for message in interaction]
                              for interaction in send_messages_list]

        self_persona_concepts = extract_concepts(self.persona_cosplay, 50)
        partner_persona_concepts = extract_concepts(partner_persona, 50)

        self_persona_set = concept_set(self_persona_concepts, self.device)
        partner_persona_set = concept_set(partner_persona_concepts, self.device)

        batch_size = len(send_messages_list[0])
        num_turn = len(send_messages_list)

        common_ground_rewards = [[0 for _ in range(num_turn)] for _ in range(batch_size)]
        recall_scores = [[0 for _ in range(num_turn)] for _ in range(batch_size)]
        future_set = [torch.zeros(2680).to(self.device) for _ in range(batch_size)]
        for idx_turn, receive_messages, send_messages in zip(
                reversed(range(num_turn)), reversed(receive_messages_list), reversed(send_messages_list)):
            for idx_batch, receive_message, send_message in zip(range(batch_size), receive_messages, send_messages):
                current_turn_concepts = extract_concepts(send_message + ' ' + receive_message, 50)
                current_turn_set = concept_set(current_turn_concepts, self.device)
                future_set[idx_batch] = set_union_operation(current_turn_set, future_set[idx_batch])

                COMMON_GROUND_REWARD()
                d_FS = set_dist_operation(future_set[idx_batch], self_persona_set, self.kw_graph_distance_matrix)
                d_FP = set_dist_operation(future_set[idx_batch], partner_persona_set, self.kw_graph_distance_matrix)
                common_ground_rewards[idx_batch][idx_turn] += (1 / d_FS + 1 / d_FP)

                PERSONA_RECALL_SCORE()
                both_persona_set = set_union_operation(self_persona_set, partner_persona_set)
                recall_score = persona_recall_score(both_persona_set, future_set[idx_batch],
                                                    self.kw_graph_distance_matrix, r)
                recall_scores[idx_batch][idx_turn] += recall_score / (num_turn - idx_turn + 1)

        common_ground_rewards = torch.tensor(common_ground_rewards).cpu().numpy()
        recall_scores = torch.tensor(recall_scores).cpu().numpy()
        return common_ground_rewards, recall_scores

    def load(self, path, override=True):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if not os.path.isfile(path + '.opt'):
            # backwards compatible to old models
            self.opt = self.override_opt(states['opt'])
            # save .opt file to make compatible
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path + ".opt", 'rb') as handle:
                opt_pickle = pickle.load(handle)
                if override:
                    self.opt = self.override_opt(opt_pickle)
                else:
                    return states, opt_pickle
        return states

    def save(self, path=None):
        """Save model parameters if model_file is set.
        """
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            optimizer = getattr(self, 'cosplay_optimizer', None)
            component = getattr(self, 'cosplay')
            model = {'model': component.state_dict(), 'longest_label': component.longest_label,
                     'optimizer': optimizer.state_dict() if optimizer is not None else None,
                     'optimizer_type': self.opt['optimizer_cosplay'], 'opt': self.opt}

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file as json
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print('Save model at {}!'.format(path))
        else:
            print('Please specify a valid save path in `model_file` keyword.')


class mydefaultdict(defaultdict):
    """Custom defaultdict which overrides defaults requested by the get
    function with the default factory.
    """

    def get(self, key, default=None):
        # override default from "get" (like "__getitem__" already is)
        return super().get(key, self.default_factory())

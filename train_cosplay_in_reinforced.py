"""
Double Dual Training
1 Policy Gradient Training with Supervised Pre-trained Cosplay
- Training in Self-play World
- Evaluate in convai2:self
"""
import os
import random
import torch

from scripts.train_model_selfplay import setup_args as setup_args_dict, TrainLoop

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IS_ORIGINAL = True

MUTUAL_BENEFIT, COMMON_GROUND, LM = 1., 1., 1.

BEAM_SIZE = 2
MODEL_NAME = 'cosplay'
RECALL_R = 0.2
BASE_DIR = '/apdcephfs/private_chencxu/taiji_outputs/cosplay/models/supervised'
MODEL_DIR = '/apdcephfs/private_chencxu/taiji_outputs/cosplay/models/reinforced'
DATA_DIR = '/apdcephfs/private_chencxu/taiji_inputs/cosplay/data'


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)


def setup_args():
    parser = setup_args_dict()
    encode_max_seq_len = 256
    decode_max_seq_len = 24

    if IS_ORIGINAL:
        cosplay_basic = 'cosplay_base'
        exp_task = 'tasks.convai2.agents:OriginalTeacher,tasks.convai2.agents:OriginalPersonaTeacher'
        exp_eval_task = 'tasks.convai2cosplay.agents:SelfOriginalTeacher:no_cands'
    else:
        cosplay_basic = 'cosplay_base'
        exp_task = 'tasks.convai2.agents:RevisedTeacher,tasks.convai2.agents:RevisedPersonaTeacher'
        exp_eval_task = 'tasks.convai2cosplay.agents:SelfRevisedTeacher:no_cands'

    exp_name = MODEL_NAME
    validation_max = -1
    train_display = 300

    # exp_name = 'DEBUG'
    parser.set_defaults(
        # idea add ================
        weights=[MUTUAL_BENEFIT, COMMON_GROUND, LM],
        recall_r=RECALL_R,
        # =======================
        download_path='{}/downloads'.format(DATA_DIR),
        datapath=DATA_DIR,
        exp=exp_name,  # name for experiment
        task=exp_task,
        personapath=MODEL_DIR,
        max_turn=3,
        evaltask=exp_eval_task,
        # TODO: now we must keep batch size equal because the world share the same agent
        evalbatchsize=6,
        batchsize=6,
        dict_lower=True,
        dict_include_valid=True,
        # not load from checkpoint
        dict_maxexs=1,
        dict_tokenizer='split',  # Important!
        datatype='train',
        history_replies='label_else_model',
        # model configuration
        model='agents.cosplay_rl.cosplay_rl:CosplayRLAgent',
        model_file='{}/{}.model'.format(MODEL_DIR, exp_name),
        lr=1e-6,
        gpt_lr=1e-6,
        # world sample configuration
        top_k=20,
        beam_size=2,
        gradient_clip=1.0,
        encode_max_seq_len=encode_max_seq_len,
        decode_max_seq_len=decode_max_seq_len,
        # supervised cosplay initialization
        init_model_cosplay='{}/{}.model'.format(BASE_DIR, cosplay_basic),
        rnn_class_cosplay='lstm',
        lookuptable_transmitter='enc_dec',
        embedding_type_transmitter='glove_fixed',
        optimizer_step=-1,
        init_model_coherent='{}/{}.model'.format(BASE_DIR, cosplay_basic),
        # validation configuration
        validation_max_exs=validation_max,  # -1
        validation_every_n_secs=2400,  # 90
        train_display_every_n_secs=train_display,
        validation_metric='f1',
        validation_metric_mode='max',
        validation_patience=5,
        log_every_n_secs=30,
        # logging configuration
        display_examples=False,
        tensorboard_log=True,
        tensorboard_tag='exp',
        report_freq=0.025,
        train_report_metrics='num_selfplay_turns,num_selfplay_episode,total_reward,reward,reward_var',
        tensorboard_metrics='ppl,reward,total_reward,f1,hits@1,reward_var,bleu'
    )
    return parser


if __name__ == '__main__':
    opt = setup_args()
    set_seed()
    TrainLoop(opt).train()

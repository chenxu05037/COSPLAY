from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

IS_ORIGINAL = True

# ---------------------
VIS = True
CSV = None


# ---------------------

def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2cosplay.agents:SelfOriginalTeacher'
    else:
        task_name = 'tasks.convai2cosplay.agents:SelfRevisedTeacher'
    return task_name


def setup_trained_weights():
    if IS_ORIGINAL:
        weights_name = '/apdcephfs/private_chencxu/taiji_outputs/cosplay/models/reinforced/cosplay.model'
    else:
        weights_name = '/apdcephfs/private_chencxu/taiji_outputs/cosplay/models/reinforced/cosplay.model'
    return weights_name


def setup_args(parser=None):
    parser = base_setup_args(parser)
    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        datatype='valid',
        hide_labels=False,
        metrics='f1,bleu',
    )
    return parser


def eval_f1(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('Final F1@1: {}, BLEU:  {}'.format(report['f1'], report['bleu']))


if __name__ == '__main__':
    parser = setup_args()
    model_name = setup_trained_weights()
    parser.set_params(
        datapath='/apdcephfs/private_chencxu/taiji_inputs/cosplay/data',
        model='agents.cosplay.cosplay:TransformerAgent',
        model_file=model_name,
        gpu=0,
        batchsize=1,
        beam_size=2,
        vis=VIS,
        visualization=True
    )
    opt = parser.parse_args(print_args=False)
    eval_f1(opt, print_parser=parser)

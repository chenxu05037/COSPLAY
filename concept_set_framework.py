from nltk.util import ngrams
import nltk
import random
from tqdm import tqdm

from code_structure import SET_EXPANSION

nltk.data.path.append('/apdcephfs/private_chencxu/taiji_inputs/cosplay/data/nltk_data')
from nltk.stem import WordNetLemmatizer
import pickle
import torch

_lemmatizer = WordNetLemmatizer()
name_list = ['concept2id', 'id2concept', 'node2id', 'word2id', 'CN_hopk_graph_dict']

pkl_list = []
for name in name_list:
    with open('/apdcephfs/private_chencxu/taiji_inputs/cosplay/data/concept_set_framework/{}.pkl'.format(name),
              "rb") as f:
        pkl_list.append(pickle.load(f))

concept2id, id2concept, node2id, word2id, CN_hopk_graph_dict = pkl_list


# idea interface
def prepare_example_persona_kws(history, persona_str):
    if history and 'persona_kws' in history:
        return history['persona_kws']
    else:
        return extract_concepts(persona_str, 30)


def prepare_batch_persona_concept_mask(obs, device):
    batch_persona_kws = torch.tensor([o['persona_kws'] for o in obs if len(o['text2vec']) > 0]).to(device)
    mask = torch.zeros(len(concept2id)).to(device).unsqueeze(0).expand(len(batch_persona_kws), -1)
    batch_persona_kws_mask = mask.scatter(dim=1, index=batch_persona_kws, src=torch.ones_like(mask))
    batch_persona_kws_mask[:, 0:2] = 0
    return batch_persona_kws_mask


def create_concept_dist_matrix(path, device, debug=False):
    MAX = 100
    concept_dist_matrix = torch.ones((len(concept2id), len(concept2id))).to(device) * -1
    concept_distance_dict = load_pickle(path)
    for node1, node2 in tqdm(concept_distance_dict.keys(), desc='creating concept distance matrix'):
        concept_dist_matrix[concept2id[node1], concept2id[node2]] = concept_distance_dict[(node1, node2)]
    concept_dist_matrix[torch.isinf(concept_dist_matrix)] = -1.
    max_distance = concept_dist_matrix.max().item()
    concept_dist_matrix = torch.where(concept_dist_matrix.eq(-1),
                                      torch.ones_like(concept_dist_matrix) * MAX,
                                      concept_dist_matrix)
    min_distance = concept_dist_matrix.view(-1).topk(2680, largest=False)[0].unique()[1].item()
    concept_dist_matrix = torch.where(concept_dist_matrix.eq(0),
                                      torch.ones_like(concept_dist_matrix) * min_distance,
                                      concept_dist_matrix)

    output_pkl = {'matrix': concept_dist_matrix, 'max': max_distance, 'min': min_distance}
    pkl_path = '/'.join(path.split('/')[:-1]) + '/concept_dist_matrix.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(output_pkl, f, protocol=4)

    return output_pkl


def load_concept_dist_matrix(pkl_file):
    with open(pkl_file, 'rb') as pkl_in:
        ret = pickle.load(pkl_in)
    return ret


def visualize_topk_nodes_with_values(tensor, vocab, k=10, concept=False, matrix=False):
    visualization = ''
    if matrix is False:
        idx = tensor.topk(k)[1].tolist()
        if concept:
            concepts = [vocab[i] for i in idx]
            values = tensor.topk(10)[0].tolist()
            visualization += ' '.join(
                ['{:>6}'.format(c) + '(' + str('{:.3f}'.format(v)) + ')' for c, v in zip(concepts, values)])
    else:
        idx = torch.topk(tensor, 5, dim=-1)[1].transpose(0, 1).tolist()
        values = tensor.topk(5)[0].transpose(0, 1).tolist()
        for i, v in zip(idx, values):
            words = vocab.vec2words(i)
            line = ' '.join(['{:>8}'.format(word) + '(' + str('{:.3f}'.format(prob)) + ')'
                             for word, prob in zip(words, v)])
            visualization += (line + '\n')
    return visualization


def normalization(scores):
    min_score = scores.min()
    max_score = scores.max()
    diff_reward = max_score - min_score + 1e-6
    return (scores - min_score) / diff_reward


def standardization(rewards):
    reward_baseline = rewards.mean(axis=0, keepdims=True)
    reward_list = rewards - reward_baseline
    return reward_list


def concept_set(concepts, device):
    # convert concepts into concept set
    concept_set = torch.scatter(input=torch.zeros(2680).to(device), dim=-1, index=torch.tensor(concepts).to(device),
                                src=torch.ones_like(torch.tensor(concepts, dtype=torch.float).to(device)))
    concept_set[0] = 0
    return concept_set


def set_dist_operation(a, b, dist_matrix):
    matrix = dist_matrix['matrix']
    dist = a.unsqueeze(0).mm(matrix).mm(b.unsqueeze(1)) / (a.sum() * b.sum())
    dist = dist.item()
    if dist.isnan():
        dist = dist_matrix['max']
    return dist


def set_union_operation(a, b):
    return torch.logical_or(a, b).type(torch.float)


def set_inter_operation(a, b, dist_matrix, r):
    a = a.type(torch.bool)
    b = b.type(torch.bool)
    M = dist_matrix['matrix'][a][:, b]
    if 0 == M.size(0) * M.size(1):
        inter_set = torch.zeros_like(a)
    else:
        inter_set = torch.where(M < r, torch.ones_like(M), torch.zeros_like(M)).sum(0).clamp(0, 1)
    return inter_set


def set_expa_operation(a, dist_matrix, topk=None, concept2words_map=None):
    matrix = dist_matrix['matrix']
    max = dist_matrix['max']

    to_a_dist = matrix * a.unsqueeze(1)  # [bs, 2680]
    # mask 后产生 0， 需要将其替换为 max
    to_a_dist = torch.where(to_a_dist.eq(0), torch.ones_like(to_a_dist) * 100, to_a_dist)

    to_a_dist = max - to_a_dist.min(dim=-1)[0]

    # 去掉 GPT 词典中没有的 concept for attention calculation
    to_a_dist = to_a_dist * concept2words_map.sum(-1).ne(0)

    to_a_dist = torch.where(to_a_dist.eq(0), torch.ones_like(to_a_dist) * -1e10, to_a_dist)
    to_a_dist = top_k_logits(to_a_dist, topk)

    # 精确 topk 个数，for attention calculation
    expanded_a = torch.scatter(input=torch.zeros_like(to_a_dist), index=to_a_dist.topk(topk)[1],
                               src=torch.ones_like(to_a_dist), dim=-1)

    return expanded_a


def persona_recall_score(persona_set, future_set, dist_matrix, r=0.2):
    inter_set = set_inter_operation(future_set, persona_set, dist_matrix, r=r)
    recall_score = inter_set.sum() / sum(persona_set).clamp(0.1)
    return recall_score.item()


def have_concepts_in(common_ground_one_turn):
    return common_ground_one_turn.sum() > 1


def cal_context_set(context_concepts, device, lower_bound=0):
    batch_size = context_concepts.size(0)
    context_concept_pool = torch.scatter(input=torch.zeros([batch_size, 2680], dtype=torch.bool, device=device),
                                         src=torch.ones_like(context_concepts, dtype=torch.bool),
                                         index=context_concepts, dim=-1)  # [bs, 2680]
    context_concept_pool[:, 0:2] = 0

    if lower_bound is not None:
        exceed_lower_bound = (context_concept_pool.sum(-1) >= lower_bound).unsqueeze(-1)
        context_concept_pool = context_concept_pool * exceed_lower_bound

    return context_concept_pool


def cal_concept_set(opt, context_concepts, persona_set, kw_graph_distance_matrix, device, concept2words_map, k=250):
    context_lower_bound = opt.get('context_lower_bound')

    # if size of pool < lower_bound, then this pool = 0
    context_set = cal_context_set(context_concepts=context_concepts,
                                  lower_bound=context_lower_bound, device=device)
    SET_EXPANSION()
    union_set = set_union_operation(context_set, persona_set)
    SET_EXPANSION()
    expanded_set = set_expa_operation(union_set, dist_matrix=kw_graph_distance_matrix, topk=k,
                                      concept2words_map=concept2words_map)

    has_concept = expanded_set.sum(-1).clamp(0, 1).unsqueeze(-1)
    expanded_set = expanded_set * has_concept + (1 - has_concept)

    return expanded_set


def cal_lm_word_probs(logits, softmax, temperature=1.0):
    probs = softmax(logits / temperature)
    return probs


def cal_concept_word_probs(logits, final_pool, concept2words_map, softmax, temperature=1.0):
    assert len(logits.size()) == 3
    assert len(final_pool.size()) == 2
    batch_size = logits.size(0)
    output_len = logits.size(1)
    # concept_probs = (jump_probs * hybrid_weights['jump'] + walk_probs * hybrid_weights['walk'])

    concept_word_probs, concept_word_mask = None, None
    if final_pool is not None:
        # [bs, 2680]
        topk_concept2words_map = (final_pool.unsqueeze(-1) * concept2words_map).view(batch_size, -1)
        # assert topk_concept2words_map.size() == (batch_size, 2680 * 7)

        # map to the word vocab
        idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1).type(torch.int64)
        concept_word_logits_mask = torch.scatter(input=torch.zeros_like(logits, dtype=torch.int64), index=idx,
                                                 src=torch.ones_like(idx), dim=-1)
        concept_word_logits_mask[:, :, 0] = 0
        concept_word_logits = logits * concept_word_logits_mask

        concept_word_logits = torch.where(concept_word_logits.eq(0), torch.ones_like(concept_word_logits) * -1e10,
                                          concept_word_logits)
        concept_word_probs = softmax(concept_word_logits / temperature)

    # topk_concept_idx = concept_probs.topk(topk)[1]
    # topk_concept_probs = concept_probs.topk(topk)[0]
    #
    # #  [bs, topk, 7]
    # topk_concept2words_map = torch.gather(input=concept2words_map.unsqueeze(0).expand(batch_size, -1, -1), dim=1,
    #                                       index=topk_concept_idx.unsqueeze(-1).expand(-1, -1, 7))
    #
    # # topk_concept_probs = torch.gather(input=concept_probs, dim=1, index=topk_concept_idx)
    # topk_concept2words_mask = topk_concept2words_map.ne(0)
    #
    # #  [bs, len, topk, 7]
    # concept_word_logits = torch.gather(lm_logits.unsqueeze(-2).expand(batch_size, output_len, topk, -1), dim=-1,
    #                                    index=topk_concept2words_map.type(torch.int64).unsqueeze(1).expand(
    #                                        batch_size, output_len, topk, -1))
    # concept_word_logits2 = concept_word_logits * topk_concept2words_mask.unsqueeze(1).expand(-1, output_len, -1, -1)

    # if use_lm_logits:
    #     # map to the word vocab
    #     idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1, -1).view(batch_size, output_len, -1).type(
    #         torch.int64)
    #     src = concept_word_logits2.view(batch_size, output_len, -1)
    #     tgt = torch.zeros_like(lm_logits)
    #     final_logits = tgt.scatter(dim=-1, index=idx, src=src)
    #     final_logits = torch.where(final_logits.eq(0), torch.ones_like(final_logits) * -1e10, final_logits)
    #     final_probs = softmax(final_logits)
    #
    # else:
    #     concept_word_logits3 = torch.where(concept_word_logits2.eq(0), torch.ones_like(concept_word_logits2) * -1e10,
    #                                        concept_word_logits2)
    #     word_probs_given_concept = softmax(concept_word_logits3)
    #     # word_probs_given_concept[:, :, 0:2] = 0
    #
    #     concept_word_probs = word_probs_given_concept * (topk_concept_probs.unsqueeze(-1).unsqueeze(1))
    #
    #     # map to the word vocab
    #     idx = topk_concept2words_map.unsqueeze(1).expand(-1, output_len, -1, -1).view(batch_size, output_len, -1).type(
    #         torch.int64)
    #     src = concept_word_probs.view(batch_size, output_len, -1)
    #     tgt = torch.zeros_like(lm_logits)
    #     final_probs = tgt.scatter(dim=-1, index=idx, src=src)

    return concept_word_probs


def cal_concept_word_probs_attention(embed, hidden, final_pool, concept2words_map, lm_word_probs, softmax, model):
    assert len(hidden.size()) == 3
    assert len(final_pool.size()) == 2

    batch_size = hidden.size(0)
    output_len = hidden.size(1)

    # [bs, 2680, 7]
    concept_words = final_pool.unsqueeze(-1) * concept2words_map

    # [bs, topk, 7]

    # print('concept2words_map: {}'.format(concept2words_map.size()))
    # print('concept_words: {}'.format(concept_words.size()))

    top_concept2word_map = concept2words_map.unsqueeze(0).expand(batch_size, -1, -1)[
        concept_words.sum(-1).gt(0).type(torch.bool)].view(
        batch_size, -1, 7)

    tempt_probs = torch.tensor(lm_word_probs)
    topk = top_concept2word_map.size(1)

    tempt_probs[:, :, 0] = 0
    top_concept_word_p = torch.gather(dim=-1,
                                      input=tempt_probs.unsqueeze(-2).expand(-1, -1, top_concept2word_map.size(-2),
                                                                             -1),
                                      index=top_concept2word_map.unsqueeze(1).expand(-1, output_len, -1, -1).type(
                                          torch.int64))
    # [bs, len, top]
    max_idx = top_concept_word_p.max(-1)[1]

    # [bs, len, top, 1] after lm prob to choose in each concept group
    concept2word_idx = torch.gather(input=top_concept2word_map.unsqueeze(1).expand(-1, output_len, -1, -1),
                                    index=max_idx.unsqueeze(-1),
                                    dim=-1).squeeze(-1)

    concept_embed = embed[concept2word_idx.type(torch.long)]

    # scores = model.linear_ec(concept_embed) + model.linear_hl(hidden).unsqueeze(-1)
    # scores = torch.bmm(concept_embed.view(-1, topk, 768), hidden.unsqueeze(-1).contiguous().view(-1, 768, 1)).view(
    #     batch_size, output_len, topk, -1).squeeze(-1)
    # shifted_hidden = torch.cat([hidden[:, 0:1, :], hidden[:, :-1, :]], dim=1)
    scores = torch.matmul(concept_embed, hidden.unsqueeze(-1)).squeeze(-1)
    # torch.matmul(concept_embed, hidden.unsqueeze(-1))
    # scores = torch.matmul(model.w(concept_embed), hidden.unsqueeze(-1)).squeeze(-1)
    # weighted_sum_concept_embed = (softmax(scores).unsqueeze(-1) * concept_embed).sum(dim=-2)
    weighted_sum_concept_embed = torch.matmul(softmax(scores).unsqueeze(-2), concept_embed).squeeze(-2)
    # weighted_sum_concept_embed = (softmax(scores).unsqueeze(-1) * concept_embed).sum(dim=-2)

    probs = torch.scatter(input=torch.zeros_like(lm_word_probs), src=softmax(scores), index=concept2word_idx, dim=-1)
    return probs, weighted_sum_concept_embed


def cal_hybrid_word_probs(lm_word_probs, concept_word_probs, gate, lm_mask, ablation=False):
    # jump or walk [10, 2680]
    assert len(gate.size()) == 3
    assert len(lm_word_probs.size()) == 3

    # for gate only optimize examples with concepts in the response
    if ablation:
        hybrid_probs = lm_word_probs * (1 - torch.zeros_like(gate)) + torch.zeros_like(gate) * concept_word_probs
    elif lm_mask is not None:
        hybrid_probs = lm_word_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(
            1) * concept_word_probs
    else:
        hybrid_probs = lm_word_probs * (1 - gate) + gate * concept_word_probs
    return hybrid_probs


def cal_word2concept_map(dict, device):
    map = [0] * 40516
    tokenizer = dict.tokenizer
    keys = tokenizer.decoder.keys()
    count = 0
    for idx in keys:
        word = tokenizer.decode([idx])
        if word in concept2id:
            map[idx] = concept2id[word]
            count += 1
        else:
            basic_form_word = kw_format([word])[0]
            if basic_form_word in concept2id:
                map[idx] = concept2id[basic_form_word]
                count += 1
            else:
                map[idx] = 0
    return torch.tensor(map).to(device)


def cal_concept2word_map(word_concept_map, device):
    lists = [[0.] * 7]
    for i in range(1, 2680):
        concept2words_idx = torch.where(word_concept_map.eq(i))[0]
        lists.append(torch.cat([concept2words_idx, torch.zeros(7 - len(concept2words_idx)).to(device)]).tolist())
    concept2word_map = torch.tensor(lists, dtype=torch.int64)
    return concept2word_map.to(device)
    # # list = [word_concept_map.eq(i).sum().item() for i in range(2680)]
    # # max = torch.tensor(list).topk(10)[0]
    # # concept_word_mask = torch.cat(
    # #     [word_concept_map.eq(concept_id).unsqueeze(0) + 0 for concept_id in range(len(keyword2id))], dim=0)
    # concept_word_mask[0] = 0
    # assert (word_concept_map > 0).sum() == concept_word_mask.sum()
    # return concept_word_mask


## one example for kw model
def prepare_example_for_kw_model(history, text, dict):
    context, last_two_utters = process_context(history, text, dict)
    last_two_utters_keywords = extract_concepts(context, 20)
    last_two_utters_concepts = extract_concepts_grams(context, node2id, 30, dict)
    return last_two_utters, last_two_utters_keywords, last_two_utters_concepts


## one batch for kw model
def prepare_batch_for_kw_model(obs, device):
    inputs_for_kw_model = {}

    for_kw_models = [x['kw_model'] for x in obs if len(x['text2vec']) > 0]
    itr = zip(*for_kw_models)
    try:
        batch_context = torch.tensor(next(itr)).to(device)
        batch_context_keywords = torch.tensor(next(itr)).to(device)
        batch_context_concepts = torch.tensor(next(itr)).to(device)
        CN_hopk_edge_index = torch.tensor(CN_hopk_graph_dict["edge_index"]).transpose(0, 1).to(device)  # (2, num_edges)

        inputs_for_kw_model['batch_context'] = batch_context
        inputs_for_kw_model['batch_context_keywords'] = batch_context_keywords
        inputs_for_kw_model['batch_context_concepts'] = batch_context_concepts
        inputs_for_kw_model['CN_hopk_edge_index'] = CN_hopk_edge_index
    except:
        inputs_for_kw_model = None
    return inputs_for_kw_model


def inputs_for_gate_module(tgt_seq, word2concept_map, concept_set):
    # len_gate_label = len(src) + len(tgt)
    bs = tgt_seq.size(0)
    concept_set = torch.gather(input=concept_set, index=word2concept_map.unsqueeze(0).expand(bs, -1), dim=-1)

    gate_label = tgt_seq.clone()
    gate_label[gate_label == 0] = -1

    tail = gate_label * gate_label.eq(-1)

    gate_label = torch.gather(input=concept_set, index=gate_label * gate_label.ne(-1), dim=-1) + 0
    gate_label = gate_label + tail
    # gate_label[gate_label != -1] = (pool.gather(-1, gate_label[gate_label != -1])) + 0

    # gate_label[gate_label != -1] = (word2concept_map.gather(0, gate_label[gate_label != -1]) != 0) + 0

    gate_mask = (gate_label != -1) + 0
    gate_label.masked_fill_(gate_label == -1, 0)

    lm_mask = (gate_label.sum(1) != 0).float().unsqueeze(1)
    gate_mask = lm_mask.expand_as(gate_label) * gate_mask

    gate = {
        'lm_mask': lm_mask,
        'gate_label': gate_label,
        'gate_mask': gate_mask
    }
    return gate


# Others
def get_keyword_mask_matrix(device):
    keyword_mask_matrix = torch.from_numpy(
        CN_hopk_graph_dict["edge_mask"]).float().to(device)  # numpy array of (keyword_vocab_size, keyword_vocab_size)
    print("building keyword mask matrix...")
    keyword_vocab_size = len(concept2id)
    keyword_mask_matrix[torch.arange(keyword_vocab_size).to(device), torch.arange(keyword_vocab_size).to(
        device)] = 0  # remove self loop
    return keyword_mask_matrix


def process_context(history, text, dict):
    context = ''
    if text is None or text == '__silence__':
        minus_one = [0] * 30
    else:
        context += text
        minus_one = dict.split_tokenize(context)
        minus_one = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_one]
        minus_one = pad_sentence(minus_one, 30, word2id["<pad>"])

    if history is None or history == '__silence__':
        # if history['labels']:
        #     history_text = history['labels'][0]
        # else:
        #     history_text = ''
        # dialog = np.array(history['dialog'])
        # idx_beg = np.where(dialog == 40478)[0][-1].item()
        # idx_end = np.where(dialog == 40479)[0][-1].item()
        # history_text = dict.tokenizer.decode(dialog[idx_beg + 1: idx_end])
        minus_two = [0] * 30
    else:
        context = history + ' ' + context
        minus_two = dict.split_tokenize(history)
        minus_two = [word2id[w] if w in word2id else word2id["<unk>"] for w in minus_two]
        minus_two = pad_sentence(minus_two, 30, word2id["<pad>"])

    return context, [minus_two, minus_one]


def extract_concepts(context, max_sent_len):
    rest = []
    concept_id = []
    tokenized = concept_tokenize(context)

    for w in tokenized:
        if w in concept2id:
            concept_id.append(concept2id[w])
            continue
        rest.append(w)

    basic = concept_to_basic(rest)

    for w in basic:
        if w in concept2id:
            concept_id.append(concept2id[w])

    concept_id = pad_sentence(concept_id, max_sent_len, concept2id["<pad>"])

    return concept_id


def extract_concepts_grams(context, node2id, max_sent_len, dict):
    context = dict.split_tokenize(context)
    utter_concepts = []
    all_utter_ngrams = []
    for n in range(5, 0, -1):
        all_utter_ngrams.extend(ngrams(context, n))
    for w in all_utter_ngrams:
        w = "_".join(w)
        if w in node2id and not any([w in ngram for ngram in utter_concepts]):
            utter_concepts.append(w)
    utter_concepts = [node2id[w] for w in utter_concepts]
    utter_concepts = pad_sentence(utter_concepts, max_sent_len, node2id["<pad>"])
    return utter_concepts


def pad_sentence(sent, max_sent_len, pad_token):
    if len(sent) >= max_sent_len:
        return sent[:max_sent_len]
    else:
        return sent + (max_sent_len - len(sent)) * [pad_token]


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def concept_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower])


def concept_to_basic(string):
    return tokenize(string, [pos_tag, to_basic_form])


def kw_format(string):
    return tokenize(string, [pos_tag, to_basic_form])


def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def walk_logits(logits, k):
    """
    Masks everything but the neighbors of the context concepts
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    # d = 2 / ((1 + k) * k)
    # p = [i * d for i in range(1, k + 1)]
    # pp = torch.tensor(p).unsqueeze(0).expand(logits.size(0), -1)
    # idx = torch.topk(logits, k)[1]
    # probs = torch.scatter(input=torch.zeros_like(logits), dim=-1, index=idx, src=pp)
    logits = torch.where(logits == 0.0, torch.ones_like(logits) * -1e10, logits)
    return logits


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[..., -1:]
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def visualize_samples(data_for_visualization, dict, valid_inds, observations):
    i = random.randint(0, len(data_for_visualization) - 1)
    prediction = data_for_visualization[i]['prediction']
    final_pool = data_for_visualization[i]['final_pool']
    concept_word_probs = data_for_visualization[i]['concept_word_probs']
    hybrid_word_probs = data_for_visualization[i]['hybrid_word_probs']
    lm_word_probs = data_for_visualization[i]['lm_word_probs']
    gate = data_for_visualization[i]['gate'].squeeze(-1).tolist()

    #  construct visulization strings
    line_outputs = dict.vec2words(prediction.tolist())
    vis_prediction = ' '.join(['{:>5}'.format(i) for i in line_outputs])
    vis_lm_word_probs = visualize_topk_nodes_with_values(lm_word_probs, dict, k=5, concept=False, matrix=True)
    vis_concept_word_probs = visualize_topk_nodes_with_values(concept_word_probs, dict, k=5, concept=False,
                                                              matrix=True)
    vis_hybrid_word_probs = visualize_topk_nodes_with_values(hybrid_word_probs, dict, k=5, concept=False,
                                                             matrix=True)
    print('=' * 150)
    print('Text | {}'.format(observations[valid_inds[i]]['text']))
    print('Concept Set | {}'.format(len(final_pool)))
    print('Generation | {}'.format(vis_prediction))
    gate_str = ' '.join(['{:>7}'.format(w) + '(' + str('{:.4f}'.format(g)) + ')' for w, g in
                         zip(line_outputs, gate)])
    print('Gate | {}'.format(gate_str))
    print('Set Attention | {}'.format(vis_concept_word_probs))
    print('Word Likelihood | {}'.format(vis_hybrid_word_probs))

    return

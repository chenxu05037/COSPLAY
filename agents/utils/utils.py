import numpy as np


def _calculate_expect_softmax_loss(score_view, ys):
    denominator = np.exp(score_view).sum(-1)
    numerator = []
    for i, y in enumerate(ys):
        numerator.append(np.exp(score_view[i][y]).reshape(1, ))
    numerator = np.concatenate(numerator, axis=0)
    out = numerator / denominator
    return np.mean(- np.log(out))


def padding(_xs, max_1d_size, max_2d_size, null_idx):
    for _ in range(max_1d_size - len(_xs)):
        _xs.append([])
    for x in _xs:
        for _ in range(max_2d_size - len(x)):
            x.append(null_idx)


def padding4d(_xs, max_1d_size, max_2d_size, max_3d_size, null_idx):
    for _ in range(max_1d_size - len(_xs)):
        _xs.append([])
    for x in _xs:
        for _ in range(max_2d_size - len(x)):
            x.append([])
    for x in _xs:
        for y in x:
            for _ in range(max_3d_size - len(y)):
                y.append(null_idx)


def split_pad_vector(xs, separator, null_idx):
    """
    Use the splitor to split the sentences.

    spliter is the value that represents END TOKEN
    :param x: input
    :param separator: the required seperator
    :return: a list of dialogs after splitting and padding
    """

    def split(x):
        _xs = []
        temp_x = []
        for _x in x:
            if _x == separator:
                _xs.append(temp_x)
                temp_x = []
                continue
            if _x != null_idx:
                temp_x.append(_x)
        if len(temp_x):
            _xs.append(temp_x)
        return _xs

    def get_max_words_size(_xs):
        max_size = 0
        for agent in _xs:
            for dialog in agent:
                if len(dialog) > max_size:
                    max_size = len(dialog)
        return max_size

    xs = [split(x) for x in xs]
    max_turn_size = max((len(x) for x in xs))
    max_words_size = get_max_words_size(xs)
    for agent in xs:
        padding(agent, max_turn_size, max_words_size, null_idx)
    return xs


def split_pad_vector_for_bug(xs, separator, null_idx):
    """
    Use the splitor to split the sentences.

    spliter is the value that represents END TOKEN
    :param x: input
    :param separator: the required seperator
    :return: a list of dialogs after splitting and padding
    """

    # coherent send
    def split_40483(x):
        _xs = []
        temp_x = []
        for _x in x:
            if _x == 40483:
                if temp_x[-1] == 40484:
                    _xs.append(temp_x)
                    temp_x = []
                continue
            if _x != null_idx:
                temp_x.append(_x)
        if len(temp_x):
            _xs.append(temp_x)
        return _xs

    # language model send
    def split_40484(x):
        _xs = []
        temp_x = []
        for _x in x:
            if _x in [40483, 40478, 40479, 40480, 40481]:
                continue
            if _x == separator:
                _xs.append(temp_x)
                temp_x = []
                continue
            if _x != null_idx:
                temp_x.append(_x)
        if len(temp_x):
            _xs.append(temp_x)
        return _xs

    def get_max_words_size(_xs):
        max_size = 0
        for agent in _xs:
            for dialog in agent:
                if len(dialog) > max_size:
                    max_size = len(dialog)
        return max_size

    if 40483 == separator:
        xs = [split_40483(x) for x in xs]
    else:
        xs = [split_40484(x) for x in xs]
    max_turn_size = max((len(x) for x in xs))
    max_words_size = get_max_words_size(xs)
    for agent in xs:
        padding(agent, max_turn_size, max_words_size, null_idx)
    return xs


def reverse_padding(xs, PAD_IDX=0):
    """
    Move the PAD_IDX in front of the dialog

    :param xs: input dialogs, which are encoded by dictionary,
    :param PAD_IDX: the index of the __NULL__

    Examples
    --------
    >>> xs = [[3, 1, 2, 0, 0],
    ...       [2, 1, 4, 0, 0]]
    >>> reverse_padding(xs, 0)
    [[0, 0, 3, 1, 2],
     [0, 0, 2, 1, 4]]

    """
    if not isinstance(xs, list):
        xs = [[x for x in ex] for ex in xs]

    ans = []
    if len(xs) == 0:
        return xs
    n = len(xs[0])
    for line in xs:
        end_idx = n - 1
        for end_idx in range(n - 1, -1, -1):
            if line[end_idx] != PAD_IDX:
                break
        end_idx += 1
        padding_num = n - end_idx
        new_line = [PAD_IDX] * padding_num + line[:end_idx]
        ans.append(new_line)
    return ans

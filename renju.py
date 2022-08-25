import numpy as np

BOARD_SIZE = 9
ACTION_SPACE = BOARD_SIZE * BOARD_SIZE * 2
C = BOARD_SIZE // 2
END_TH = 50
INPUT_CH = 2


def get_init():
    return np.zeros((2, BOARD_SIZE, BOARD_SIZE))


def b2s(board):
    s = ""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[0, i, j] == 1:
                s += "o"
            elif board[1, i, j] == 1:
                s += "x"
            else:
                s += "-"
    return s


def get_valid_actions(board):
    """
    Parameters:
    -----
    board: numpy.array(2, 15, 15)

    Returns:
    ------
    numpy.array(*,)
        valid_actions
    """
    action_mask = get_action_mask(board)
    return np.where(action_mask.ravel() > 0)[0]


def get_action_mask(board):
    """
    Paramters:
    -----
    board: numpy.array(2, 15, 15)

    Returns:
    -----
    numpy.array(2, 15, 15)
        action_mask
    """
    piece = int(board.sum() % 2)
    if board.sum() > 5:
        action_mask = np.zeros((2, BOARD_SIZE, BOARD_SIZE))
        action_mask[piece, :, :] = 1
    else:
        mode = int(board.sum())
        action_mask = np.zeros((2, BOARD_SIZE, BOARD_SIZE))
        # print(C, mode, piece)
        action_mask[piece, C - mode: C + 1 + mode, C - mode: C + 1 + mode] = 1
    action_mask = action_mask - np.sum(board, axis=0, keepdims=True)
    action_mask = np.clip(action_mask, 0, 1)
    # print(action_mask)
    return np.clip(action_mask, 0, 1)


def get_next_state(board, action):
    """
    Parameters
    -----
    board: numpy.array(2, 15, 15)
    action: int 0~449

    Returns
    -----
    (next_board, done, iswin)
    next_board: numpy.array(2, 15, 15)
    done: bool
    iswin: bool
    """
    piece = int(board.sum() % 2)
    assert action >= 0 and action <= ACTION_SPACE - 1, f"action: {action}"
    assert action // (BOARD_SIZE * BOARD_SIZE) == piece, f"action: {action}, piece: {piece}"
    next_board = board.copy()
    action = action % (BOARD_SIZE * BOARD_SIZE)
    i, j = action // BOARD_SIZE, action % BOARD_SIZE
    next_board[piece, i, j] = 1
    if check(board, [i, j], piece):
        return (next_board, True, True)
    if next_board.sum() > END_TH:
        return (next_board, True, False)
    return (next_board, False, False)


def check(board, action, piece):
    """
    action:  [h, w]
    """
    vecs = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
    pos = action

    for vec in vecs:
        count = 0
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() >= BOARD_SIZE or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        vec *= -1
        pos_ = pos
        while True:
            pos_ = pos_ + vec
            if pos_.max() >= BOARD_SIZE or pos_.min() < 0:
                break
            if board[piece, pos_[0], pos_[1]] == 1:
                count += 1
            else:
                break
        if count > 3:
            return True

    return False

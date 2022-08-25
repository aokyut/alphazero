import ray
from tqdm import tqdm
from network import AlphaZeroNetwork
from buffer import ReplayBuffer, Sample
from mtcs import PVmtcs
# from board import get_initial_board, get_next_state, get_action_mask
from renju import get_init, get_next_state, get_action_mask, ACTION_SPACE

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import  SummaryWriter
import multiprocessing

NUM_CPU = multiprocessing.cpu_count()
BATCH_SIZE = 32
LR = 0.0001
BUFFER_SIZE = 100


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_sims, dirichlet_alpha=0.35):
    record = []

    network = AlphaZeroNetwork()
    network.load_state_dict(weights)

    mtcs = PVmtcs(network, dirichlet_alpha)
    state = get_init()

    done = False
    i = 0
    current_player = 0
    reward = [0, 0]
    while not done:
        mtcs_policy = mtcs.search(state, num_sims)

        if i < 10:
            action = np.random.choice(
                range(ACTION_SPACE), p=mtcs_policy
            )
        else:
            action = np.where(
                np.where(np.array(mtcs_policy) == max(mtcs_policy))[0]
            )

        record.append(Sample(state, mtcs_policy, current_player, None, get_action_mask(state)))
        next_state, done, iswin = get_next_state(state, action)

        if iswin:
            reward[current_player] = 1
            reward[1 - current_player] = 1

        state = next_state
        current_player = 1 - current_player

    for sample in record:
        sample.reward = reward[0] if current_player == 0 else reward[1]
    print(record)
    return record


def main(n_parallel_selfplay=1, num_mtcs_sims=50):
    ray.init(num_cpus=NUM_CPU)
    batch_size = BATCH_SIZE

    network = AlphaZeroNetwork()

    current_weights = ray.put(network.state_dict())
    replay = ReplayBuffer(buffer_size=BUFFER_SIZE)
    optimizer = optim.SGD(network.parameters(), lr=LR)

    work_in_progress = [
        selfplay.remote(current_weights, num_mtcs_sims)
        for _ in range(n_parallel_selfplay)
    ]

    n = 0

    while n < 10000:
        for _ in tqdm(range(5)):
            finished, work_in_progress = ray.wait(work_in_progress, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            work_in_progress.extend([
                selfplay.remote(current_weights, num_mtcs_sims)
            ])
            n += 1

        num_iters = 5 * (len(replay) // batch_size)
        print(num_iters, len(replay), batch_size)
        network.train()
        for i in tqdm(range(num_iters)):
            states, masks, mtcs_policy, reward = replay.get_minibatch(batch_size=batch_size)

            p, v = network(states, masks)

            value_loss = F.mse_loss(v, reward)

            policy_loss = - mtcs_policy * torch.log(p + 0.0001)
            policy_loss = torch.sum(policy_loss)

            loss = torch.mean(policy_loss + value_loss)

            print(loss, policy_loss, value_loss)

            optimizer.zero_grad()
            loss.backward()
        network.eval()

        current_weights = ray.put(network.state_dict())


if __name__ == "__main__":
    main()

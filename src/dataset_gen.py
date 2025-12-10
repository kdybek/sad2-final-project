from boolean_network import BN
import random
import networkx as nx
from typing import Any
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


def generate_random_functions(list_of_nodes: list[str]) -> list[str]:
    """
    Generate random Boolean functions.

    Args:
        list_of_nodes (list[str]): List of node names.

    Returns:
        list[str]: List of randomly generated Boolean functions.
    """
    MAX_PARENTS = 3
    num_nodes = len(list_of_nodes)
    functions = []

    for _ in range(num_nodes):
        k = random.randint(0, MAX_PARENTS)
        if k == 0:
            functions.append(random.choice(['0', '1']))
            continue

        parents = random.sample(list_of_nodes, k)

        # Randomly negate nodes
        terms = []
        for v in parents:
            if random.random() < 0.5:
                terms.append(f'~{v}')
            else:
                terms.append(v)

        # Combine using random operators
        expr = terms[0]
        for t in terms[1:]:
            op = random.choice([' & ', ' | '])
            expr = f'({expr}{op}{t})'

        functions.append(expr)

    return functions


def generate_trajectories(
    state_transition_system: nx.DiGraph,
    attracting_states: set[str],
    num_traj: int,
    traj_len: int,
    step: int
) -> tuple[list[list[str]], float]:
    """
    Generate trajectories from Boolean network simulations.

    Args:
        sts (nx.DiGraph): State transition system of the Boolean network.
        attracting_states (set[str]): Set of attracting states. Precalculated for efficiency.
        num_traj (int): Number of trajectories to simulate.
        traj_len (int): Length of each trajectory.
        step (int): Step size for sampling states.

    Returns:
        tuple[list[list[str]], int]: List of trajectories
            and fraction of attracting states in the dataset.
    """
    num_attracting_states = 0
    dataset_size = num_traj * traj_len
    trajs = []

    for i in range(num_traj):
        state = random.choice(list(state_transition_system.nodes))
        trajectory = []
        in_attractor = False
        for j in range(traj_len * step):
            if j % step == 0:
                if not in_attractor and state in attracting_states:
                    num_attracting_states += traj_len - len(trajectory)
                    in_attractor = True

                trajectory.append(state)

            state = random.choice(list(state_transition_system.neighbors(state)))

        trajs.append(trajectory)

    attractor_fraction = num_attracting_states / dataset_size

    return trajs, attractor_fraction


def generate_datasets_for_single_bn(bn: BN) -> list[dict[str, Any]]:
    """
    Generate datasets of Boolean network trajectories.

    Args:
        bn (BN): Boolean network instance.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing
            parameters and trajectories.
    """
    NUM_TAJS_LIST = [10, 20, 50]
    TRAJ_LEN_LIST = [20, 50, 100]
    STEP_LIST = [1, 2, 3]

    NUM_SYNC_SAMPLES = 4  # Only first state is random in sync mode so we sample less
    NUM_ASYNCH_SAMPLES = 16  # There is more randomness in async mode so we sample more

    product_iter = product(NUM_TAJS_LIST, TRAJ_LEN_LIST, STEP_LIST)
    datasets = []
    for mode in ['sync', 'async']:
        sts = bn.generate_state_transition_system(mode=mode)
        attractors = nx.attracting_components(sts)
        attracting_states = set(
            state for attractor in attractors for state in attractor)
        samples = NUM_SYNC_SAMPLES if mode == 'sync' else NUM_ASYNCH_SAMPLES
        for num_traj, traj_len, step in product_iter:
            for _ in range(samples):
                trajs, attractor_fraction = generate_trajectories(
                    state_transition_system=sts,
                    attracting_states=attracting_states,
                    num_traj=num_traj,
                    traj_len=traj_len,
                    step=step)

                data_entry = {
                    'trajectories': trajs,
                    'attractor_fraction': attractor_fraction,
                    'mode': mode,
                    'step': step,
                }
                datasets.append(data_entry)

    return datasets


def generate_big_dataset_from_random_bns(seed: int) -> list[dict[str, Any]]:
    """
    Generate a dataset from randomly generated Boolean networks.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing
            parameters and trajectories.
    """
    NUM_NODES_LIST = [5, 7, 10, 13, 16]

    random.seed(seed)

    big_dataset = []
    for num_nodes in tqdm(NUM_NODES_LIST, desc="Iterating over BNs"):
        node_names = [f'x{i}' for i in range(num_nodes)]
        functions = generate_random_functions(node_names)
        bn = BN(node_names, functions)
        datasets = generate_datasets_for_single_bn(bn)
        data_entry = {
            'list_of_nodes': node_names,
            'list_of_functions': functions,
            'datasets': datasets
        }
        big_dataset.append(data_entry)

    return big_dataset


def main() -> None:
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    big_dataset = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(generate_big_dataset_from_random_bns, seed)
            for seed in SEEDS
        ]
        for future in as_completed(futures):
            big_dataset.extend(future.result())

    with open('boolean_network_datasets.pkl', 'wb') as f:
        pickle.dump(big_dataset, f)


if __name__ == "__main__":
    main()

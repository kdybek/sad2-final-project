from boolean_network import BN
import random
import networkx as nx
from typing import Any
from itertools import product
from tqdm import tqdm


def generate_random_functions(list_of_nodes: list[str], max_parents: int) -> list[str]:
    """
    Generate random Boolean functions.

    Args:
        list_of_nodes (list[str]): List of node names.
        max_parents (int): Maximum number of parents per node.

    Returns:
        list[str]: List of randomly generated Boolean functions.
    """
    num_nodes = len(list_of_nodes)
    functions = []

    for _ in range(num_nodes):
        k = random.randint(0, max_parents)
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


def generate_grid_search_datasets_from_single_bn(
        bn: BN,
        modes: list[str],
        num_trajs_list: list[int],
        traj_len_list: list[int],
        step_list: list[int],
        num_sync_samples: int,
        num_asynch_samples: int,
) -> list[dict[str, Any]]:
    """
    Generate grid search datasets of Boolean network trajectories.

    Args:
        bn (BN): Boolean network instance.
        modes (list[str]): List of update modes for simulation.
        num_trajs_list (list[int]): List of numbers of trajectories to simulate.
        traj_len_list (list[int]): List of trajectory lengths to simulate.
        step_list (list[int]): List of step sizes for sampling states.
        num_sync_samples (int): Number of samples per parameter combination in sync mode.
        num_asynch_samples (int): Number of samples per parameter combination in async mode.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing
            parameters and trajectories.
    """
    datasets = []
    for mode in modes:
        sts = bn.generate_state_transition_system(mode=mode)
        attractors = nx.attracting_components(sts)
        attracting_states = set(
            state for attractor in attractors for state in attractor)
        samples = num_sync_samples if mode == 'sync' else num_asynch_samples
        for num_traj, traj_len, step in product(num_trajs_list, traj_len_list, step_list):
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


def generate_dataset_from_single_bn(
    bn: BN,
    mode: str,
    num_traj: int,
    traj_len: int,
    step: int
) -> dict[str, Any]:
    """
    Generate a dataset of Boolean network trajectories.

    Args:
        bn (BN): Boolean network instance.
        mode (str): Update mode for simulation.
        num_traj (int): Number of trajectories to simulate.
        traj_len (int): Length of each trajectory.
        step (int): Step size for sampling states.

    Returns:
        dict[str, Any]: Dictionary containing parameters and trajectories.
    """
    sts = bn.generate_state_transition_system(mode=mode)
    attractors = nx.attracting_components(sts)
    attracting_states = set(
        state for attractor in attractors for state in attractor)

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

    return data_entry


def generate_big_dataset_from_random_bns(
    seed: int,
    num_nodes_list: list[int],
    modes: list[str],
    num_trajs_list: list[int],
    traj_len_list: list[int],
    step_list: list[int],
    max_parents: int,
    num_sync_samples: int,
    num_asynch_samples: int
) -> list[dict[str, Any]]:
    """
    Generate a dataset from randomly generated Boolean networks.

    Args:
        seed (int): Random seed for reproducibility.
        num_nodes_list (list[int]): List of numbers of nodes for the Boolean networks.
        modes (list[str]): List of update modes for simulation.
        num_trajs_list (list[int]): List of numbers of trajectories to simulate.
        traj_len_list (list[int]): List of trajectory lengths to simulate.
        step_list (list[int]): List of step sizes for sampling states.
        max_parents (int): Maximum number of parents per node.
        num_sync_samples (int): Number of samples per parameter combination in sync mode.
        num_asynch_samples (int): Number of samples per parameter combination in async mode.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing
            parameters and trajectories.
    """
    random.seed(seed)

    big_dataset = []
    for num_nodes in tqdm(num_nodes_list, desc="Iterating over BNs"):
        list_of_nodes = [f'x{i}' for i in range(num_nodes)]
        functions = generate_random_functions(
            list_of_nodes=list_of_nodes,
            max_parents=max_parents
        )
        bn = BN(list_of_nodes, functions)
        datasets = generate_grid_search_datasets_from_single_bn(
            bn=bn,
            modes=modes,
            num_trajs_list=num_trajs_list,
            traj_len_list=traj_len_list,
            step_list=step_list,
            num_sync_samples=num_sync_samples,
            num_asynch_samples=num_asynch_samples
        )
        data_entry = {
            'list_of_nodes': list_of_nodes,
            'list_of_functions': functions,
            'nodes_readable': bn.return_indexed_edges(),
            'datasets': datasets
        }
        big_dataset.append(data_entry)

    return big_dataset

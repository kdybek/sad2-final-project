from boolean_network import BN
import random
import networkx as nx
from typing import Any
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from absl import flags, app


FLAGS = flags.FLAGS

# Maximum number of parents per node in random Boolean functions.
# This should not be edited, because it was specified in the project description.
FLAGS.DEFINE_integer('max_parents', 3, 'Maximum number of parents per node.')

# Random seeds for dataset generation.
# For each seed, a separate part of the dataset is generated.
# More seeds lead to larger datasets.
# I have done it in this way to parallelize the dataset generation.
FLAGS.DEFINE_integer('num_seeds', 8, 'Number of random seeds for dataset generation.')

# List of number of nodes for the random Boolean networks.
# For each seed, Boolean networks with these numbers of nodes are generated.
# Only the middle values should be edited.
FLAGS.DEFINE_multi_integer(
    'num_nodes_list',
    [5, 7, 10, 13, 16],
    'List of numbers of nodes for the Boolean networks.'
)

# Parameters for trajectory generation.
# For every Boolean network, datasets are generated
# for all combinations of these parameters.
# Feel free to edit these (except for MODES).
FLAGS.DEFINE_multi_string('modes', ['sync', 'async'],
                          'List of update modes for simulation.')
FLAGS.DEFINE_multi_integer(
    'num_trajs_list', [10, 20, 50], 'List of numbers of trajectories to simulate.')
FLAGS.DEFINE_multi_integer(
    'traj_len_list', [20, 50, 100], 'List of trajectory lengths to simulate.')
FLAGS.DEFINE_multi_integer(
    'step_list', [1, 2, 3], 'List of step sizes for sampling states.')

# Number of samples per parameter combination and mode.
# Multiple samples are taken, because they may differ
# in the attractor to transient state ratio.
# Feel free to edit these.
FLAGS.DEFINE_integer(
    'sync_samples', 4, 'Number of samples per parameter combination in sync mode.')
FLAGS.DEFINE_integer('asynch_samples', 16,
                     'Number of samples per parameter combination in async mode.')

FLAGS.DEFINE_integer('max_workers', 8, 'Number of parallel workers.')
FLAGS.DEFINE_string('output_file', 'boolean_network_datasets.pkl',
                    'Output file for the generated datasets.')


def generate_random_functions(list_of_nodes: list[str], max_parents: int) -> list[str]:
    """
    Generate random Boolean functions.

    Args:
        list_of_nodes (list[str]): List of node names.

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


def generate_datasets_for_single_bn(
        bn: BN,
        modes: list[str],
        num_trajs_list: list[int],
        traj_len_list: list[int],
        step_list: list[int],
        num_sync_samples: int,
        num_asynch_samples: int,
) -> list[dict[str, Any]]:
    """
    Generate datasets of Boolean network trajectories.

    Args:
        bn (BN): Boolean network instance.

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
        datasets = generate_datasets_for_single_bn(
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


def main() -> None:
    big_dataset = []
    seeds = list(range(FLAGS.num_seeds))
    with ProcessPoolExecutor(max_workers=FLAGS.max_workers) as executor:
        futures = [
            executor.submit(
                generate_big_dataset_from_random_bns,
                seed=seed,
                num_nodes_list=FLAGS.num_nodes_list,
                modes=FLAGS.modes,
                num_trajs_list=FLAGS.num_trajs_list,
                traj_len_list=FLAGS.traj_len_list,
                step_list=FLAGS.step_list,
                max_parents=FLAGS.max_parents,
                num_sync_samples=FLAGS.sync_samples,
                num_asynch_samples=FLAGS.asynch_samples
            )
            for seed in seeds
        ]
        for future in as_completed(futures):
            big_dataset.extend(future.result())

    with open(FLAGS.output_file, 'wb') as f:
        pickle.dump(big_dataset, f)


if __name__ == "__main__":
    app.run(main)

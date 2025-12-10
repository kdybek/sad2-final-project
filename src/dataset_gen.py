from boolean_network import BN
import random
import networkx as nx
from typing import Any


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
        k = random.randint(1, MAX_PARENTS)
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
    bn: BN,
    num_traj: int,
    traj_len: int,
    step: int
) -> tuple[list[list[str]], float]:
    """
    Generate trajectories from Boolean network simulations.

    Args:
        bn (BN): Boolean network instance.
        num_traj (int): Number of trajectories to simulate.
        traj_len (int): Length of each trajectory.
        step (int): Step size for sampling states.

    Returns:
        tuple[list[list[str]], int]: List of trajectories
            and fraction of attracting states in the dataset.
    """
    sts = bn.generate_state_transition_system()
    attractors = nx.attracting_components(sts)
    attracting_states = set(state for attractor in attractors for state in attractor)
    num_attracting_states = 0
    dataset_size = num_traj * traj_len
    trajs = []

    for i in range(num_traj):
        state = random.choice(list(sts.nodes))
        trajectory = []
        in_attractor = False
        for j in range(traj_len):
            if j % step == 0:
                if not in_attractor and state in attracting_states:
                    num_attracting_states += traj_len - len(trajectory)
                    in_attractor = True

                trajectory.append(state)

            state = random.choice(list(sts.neighbors(state)))

        trajs.append(trajectory)

    attractor_fraction = num_attracting_states / dataset_size

    return trajs, attractor_fraction


#TODO: More parameter variety
def generate_dataset() -> list[dict[str, Any]]:
    """
    Generate a dataset of Boolean networks and their trajectories.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing
            Boolean network parameters and trajectories.
    """
    dataset = []
    list_of_nodes = ['x0', 'x1', 'x2']
    functions = generate_random_functions(list_of_nodes=list_of_nodes)
    bn = BN(list_of_nodes, functions, mode='async')
    trajs, attractor_fraction = generate_trajectories(
        bn=bn, num_traj=5, traj_len=10, step=1)

    data_entry = {
        'list_of_nodes': list_of_nodes,
        'list_of_functions': functions,
        'trajectories': trajs,
        'attractor_fraction': attractor_fraction
    }
    dataset.append(data_entry)

    return dataset


def main() -> None:
    list_of_nodes = ['x0', 'x1', 'x2']
    functions = generate_random_functions(list_of_nodes=list_of_nodes)

    print(f'list_of_nodes={list_of_nodes}')
    print(f'functions={functions}')
    bn = BN(list_of_nodes, functions, mode='async')
    trajs = generate_trajectories(bn=bn, num_traj=5, traj_len=10, step=1)
    print(f'trajs={trajs}')
    print(f'Attractor fraction={trajs[1]:.2f}')


if __name__ == "__main__":
    main()

from boolean_network import BN
import random


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


def main() -> None:
    list_of_nodes = ['x0', 'x1', 'x2']
    functions = generate_random_functions(list_of_nodes=list_of_nodes)

    print(f'list_of_nodes={list_of_nodes}')
    print(f'functions={functions}')
    bn = BN(list_of_nodes=list_of_nodes, functions=functions, mode='async')
    bn.draw_state_transition_system()


if __name__ == "__main__":
    main()

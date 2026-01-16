import networkx as nx
import boolean as bool
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import random
import re

# Code from laboratories for SAD2 course at MIM UW.


class BN:
    """Class representing a Boolean Network (BN) model."""

    __bool_algebra = bool.BooleanAlgebra()

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------

    def dependency_edges(self) -> list[tuple[str, str]]:
        """
        Return list of edges (source_node, target_node) where source_node
        appears in the Boolean function of target_node.
        """
        edges: list[tuple[str, str]] = []
        for target_idx, fun in enumerate(self.functions):
            fun_str = str(fun)
            for source_name in self.node_names:
                # use word boundaries to avoid substrings (x1 vs x10)
                if re.search(r'\b' + re.escape(source_name) + r'\b', fun_str):
                    edges.append((source_name, self.node_names[target_idx]))
        return edges

    def __str__(self) -> str:
        """
        String summary of the Boolean Network using node indices instead of names.
        """
        # Map node names to indices
        name_to_index = {name: i for i, name in enumerate(self.node_names)}

        # Compute dependency edges using indices
        indexed_edges = []
        for target_idx, fun in enumerate(self.functions):
            fun_str = str(fun)
            for source_name, source_idx in name_to_index.items():
                if re.search(r'\b' + re.escape(source_name) + r'\b', fun_str):
                    indexed_edges.append((source_idx, target_idx))

        nodes = tuple(range(self.num_nodes))
        edges = tuple(indexed_edges)

        return f"BN(nodes={nodes}, edges={edges})"
    
    def return_indexed_edges(self):
        # Map node names to indices
        name_to_index = {name: i for i, name in enumerate(self.node_names)}

        # Compute dependency edges using indices
        indexed_edges = []
        for target_idx, fun in enumerate(self.functions):
            fun_str = str(fun)
            for source_name, source_idx in name_to_index.items():
                if re.search(r'\b' + re.escape(source_name) + r'\b', fun_str):
                    indexed_edges.append((source_idx, target_idx))

        # nodes = tuple(range(self.num_nodes))
        edges = tuple(indexed_edges)

        return edges


    def __int_to_state(self, x: int) -> tuple[int, ...]:
        """
        Convert a non-negative integer into a Boolean state (tuple of 0s and 1s).

        Args:
            x (int): State number.

        Returns:
            tuple[int, ...]: Tuple of 0s and 1s representing the Boolean network state.
        """
        binary_str = format(x, '0' + str(self.num_nodes) + 'b')
        state = [int(char) for char in binary_str]
        return tuple(state)

    @staticmethod
    def __state_to_binary_str(state: tuple[int, ...]) -> str:
        """
        Convert a Boolean state (tuple) into a binary string.

        Args:
            state (tuple[int, ...]): Tuple of 0s and 1s representing the Boolean network state.

        Returns:
            str: Binary string representation of the Boolean state.
        """
        bin_str = ''.join(str(bit) for bit in state)
        return bin_str

    @staticmethod
    def __validate_mode(mode: str) -> None:
        """
        Validate the update mode.

        Args:
            mode (str): Update mode to validate.

        Raises:
            AssertionError: If the mode is not 'sync' or 'async'.
        """
        assert mode in ('sync', 'async'), "Mode must be either 'sync' or 'async'."

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------

    def __init__(self, list_of_nodes: list[str], list_of_functions: list[str]):
        """
        Initialize the Boolean Network.

        Args:
            list_of_nodes (list[str]): List of node names.
            list_of_functions (list[str]): List of Boolean expressions for each node,
                e.g. '(x0 & ~x1) | x2', where 'x0', 'x1', and 'x2' are node names.
        """
        self.num_nodes = len(list_of_nodes)
        self.node_names = list_of_nodes

        # Create Boolean symbols for nodes
        self.list_of_nodes = [
            self.__bool_algebra.Symbol(node_name) for node_name in list_of_nodes
        ]

        # Parse Boolean functions
        self.functions = [
            self.__bool_algebra.parse(fun, simplify=True) for fun in list_of_functions
        ]

    # --------------------------------------------------------------------------
    # State transitions
    # --------------------------------------------------------------------------

    def get_neighbor_states(self, state: tuple[int, ...], mode: str) -> set[tuple[int, ...]]:
        """
        Compute all states reachable from the given state in one update.

        Args:
            state (tuple[int, ...]): Tuple of 0s and 1s representing the current state.
            mode (str): Update mode, either 'sync' for synchronous or 'async' for asynchronous.

        Returns:
            set[tuple[int, ...]]: Set of next states reachable in one step.
        """
        self.__validate_mode(mode)

        neighbor_states = []
        substitutions = {
            self.list_of_nodes[i]:
                self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE
            for i, node_value in enumerate(state)
        }

        if mode == 'sync':
            new_state = []
            for fun in self.functions:
                new_node_value = 1 if fun.subs(
                    substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                new_state.append(new_node_value)
            neighbor_states.append(new_state)

        else:  # async
            for node_index, fun in enumerate(self.functions):
                new_node_value = 1 if fun.subs(
                    substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                new_state = list(state)
                new_state[node_index] = new_node_value
                neighbor_states.append(new_state)

        return set(tuple(ns) for ns in neighbor_states)

    # --------------------------------------------------------------------------
    # State transition system
    # --------------------------------------------------------------------------

    def generate_state_transition_system(self, mode: str) -> nx.DiGraph:
        """
        Generate the state transition system (STS) of the Boolean network.

        Args:
            mode (str): Update mode, either 'sync' for synchronous or 'async' for asynchronous.

        Returns:
            nx.DiGraph: Directed graph representing the STS.
        """
        self.__validate_mode(mode)

        G = nx.DiGraph()

        # Loop over all possible states (2^n)
        for initial_state_int in range(2 ** self.num_nodes):
            initial_state = self.__int_to_state(initial_state_int)
            neighbor_states = self.get_neighbor_states(initial_state, mode)

            G.add_node(self.__state_to_binary_str(initial_state))

            # Add edges from current state to all its neighbors
            edges = [
                (self.__state_to_binary_str(initial_state),
                 self.__state_to_binary_str(ns))
                for ns in neighbor_states
            ]
            G.add_edges_from(edges)

        return G


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def draw_state_transition_system(state_transition_system: nx.DiGraph, highlight_attractors: bool = True) -> None:
    """
    Draw the state transition system.

    Args:
        state_transition_system (nx.DiGraph): The state transition system to draw.
        highlight_attractors (bool, optional): If True, states belonging to different attractors
            are drawn using distinct colors. Defaults to True.

    Returns:
        None
    """
    NON_ATTRACTOR_STATE_COLOR = 'grey'

    # Assign colors to attractors if highlighting is enabled
    if highlight_attractors:
        attractors = [attractor for attractor in nx.attracting_components(
            state_transition_system)]
        sts_nodes = list(state_transition_system.nodes)
        node_colors = [NON_ATTRACTOR_STATE_COLOR for _ in sts_nodes]

        colors = list(mcolors.CSS4_COLORS)
        for color_to_remove in ('white', NON_ATTRACTOR_STATE_COLOR):
            if color_to_remove in colors:
                colors.remove(color_to_remove)

        for attractor in attractors:
            color = random.choice(colors)
            for state in attractor:
                node_colors[sts_nodes.index(state)] = color

    
    # Draw the network
    nx.draw_networkx(
        state_transition_system,
        with_labels=True,
        pos=nx.spring_layout(state_transition_system),
        node_color=node_colors,
        font_size=8
    )

    plt.show()

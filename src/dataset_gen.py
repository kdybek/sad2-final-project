from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from absl import flags, app
from dataset_gen_utils import generate_big_dataset_from_random_bns


FLAGS = flags.FLAGS

# Maximum number of parents per node in random Boolean functions.
# This should not be edited, because it was specified in the project description.
flags.DEFINE_integer('max_parents', 3, 'Maximum number of parents per node.')

# Random seeds for dataset generation.
# For each seed, a separate part of the dataset is generated.
# More seeds lead to larger datasets.
# I have done it in this way to parallelize the dataset generation.
flags.DEFINE_integer('num_seeds', 8, 'Number of random seeds for dataset generation.')

# List of number of nodes for the random Boolean networks.
# For each seed, Boolean networks with these numbers of nodes are generated.
# Only the middle values should be edited.
flags.DEFINE_multi_integer(
    'num_nodes_list',
    [5, 7, 10, 13, 16],
    'List of numbers of nodes for the Boolean networks.'
)

# Parameters for trajectory generation.
# For every Boolean network, datasets are generated
# for all combinations of these parameters.
# Feel free to edit these (except for MODES).
flags.DEFINE_multi_string('modes', ['sync', 'async'],
                          'List of update modes for simulation.')
flags.DEFINE_multi_integer(
    'num_trajs_list', [10, 20, 50], 'List of numbers of trajectories to simulate.')
flags.DEFINE_multi_integer(
    'traj_len_list', [20, 50, 100], 'List of trajectory lengths to simulate.')
flags.DEFINE_multi_integer(
    'step_list', [1, 2, 3], 'List of step sizes for sampling states.')

# Number of samples per parameter combination and mode.
# Multiple samples are taken, because they may differ
# in the attractor to transient state ratio.
# Feel free to edit these.
flags.DEFINE_integer('sync_samples', 4,
                     'Number of samples per parameter combination in sync mode.')
flags.DEFINE_integer('asynch_samples', 16,
                     'Number of samples per parameter combination in async mode.')

flags.DEFINE_integer('max_workers', 8, 'Number of parallel workers.')
flags.DEFINE_string('output_file', 'boolean_network_datasets.pkl',
                    'Output file for the generated datasets.')


def main(argv) -> None:
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

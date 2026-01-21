from dataset_gen_utils import generate_trajectories
from parse_bnet import parse_bnet_file
from absl import flags, app
import pickle
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('bnet_path', 'biodivine-models/toll-pathway-of-drosophilia.bnet', 'Relative path to .bnet file to process (run from project root)')
flags.DEFINE_string('output_path', 'output', 'Output file path for the generated dataset pickle')


def main(argv) -> None:
    
    bnet_path = FLAGS.bnet_path
    bn = parse_bnet_file(bnet_path)

    """
    def generate_dataset_from_single_bn(
    bn: BN,
    mode: str,
    num_traj: int,
    traj_len: int,
    step: int
) -> dict[str, Any]:
    """

    # Generate datasets for the provided BN
    sts = bn.generate_state_transition_system(mode='sync')
    dataset = generate_trajectories(sts, 'sync', 10, 20, 1)
    print(f"Generated dataset with {len(dataset)} entries for BN from {bnet_path}")
    
    # Save output to a file named from the input bnet basename
    base = os.path.splitext(os.path.basename(bnet_path))[0] + '.pkl'
    output = os.path.join(FLAGS.output_path, base)
    os.makedirs(FLAGS.output_path, exist_ok=True)

    all_info = dict()
    all_info['all_trajectories'] = dataset
    all_info['edges'] = bn.return_indexed_edges()
    
    with open(output, 'wb') as f:
        pickle.dump(all_info, f)

    print(f"Saved dataset to {output}")


if __name__ == "__main__":
    app.run(main)
g
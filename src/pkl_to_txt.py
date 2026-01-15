from absl import flags, app
import pickle
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', 'output/toll-pathway-of-drosophilia.pkl', 'Input directory containing .pkl files')
flags.DEFINE_string('output_path', 'output', 'Output directory for .txt files')

def main(argv) -> None:
    input_path = FLAGS.input_path
    output_path = FLAGS.output_path
    print(f"output path{output_path}")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Project root: {project_root}")
    if os.path.isabs(output_path):
        resolved_output_path = output_path
    else:
        resolved_output_path = os.path.join(project_root, output_path)

    os.makedirs(resolved_output_path, exist_ok=True)

    # Input can be either a single .pkl file path or a directory containing exactly one .pkl
    if os.path.isfile(input_path) and input_path.endswith('.pkl'):
        pkl_file = input_path
        filename = os.path.basename(pkl_file)
    else:
        if not os.path.isdir(input_path):
            raise SystemExit(f"Input path not found or not a directory/file: {input_path}")

        pkl_files = [f for f in os.listdir(input_path) if f.endswith('.pkl')]

        if len(pkl_files) == 0:
            raise SystemExit(f"No .pkl files found in input path: {input_path}")
        if len(pkl_files) > 1:
            raise SystemExit(f"Multiple .pkl files found in input path: {input_path}.\nPlease ensure there is exactly one .pkl file in the input directory.")

        # Use the single pkl file
        filename = pkl_files[0]
        pkl_file = os.path.join(input_path, filename)

    with open(pkl_file, 'rb') as f:
        dataset = pickle.load(f)

    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_file = os.path.join(resolved_output_path, txt_filename)

    with open(txt_file, 'w') as f:
        for entry in dataset:
            f.write(f"{entry}\n")

    print(f"Converted {pkl_file} to {txt_file}")

if __name__ == "__main__":
    app.run(main)

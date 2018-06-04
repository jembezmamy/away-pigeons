import subprocess
import yaml

config          = yaml.safe_load(open("config.yml"))
base_path       = config['training']['storage_path']
steps           = config['training']['steps']
architecture    = config['training']['architecture']

# cmd = ("python3 vendor/tensorflow-for-poets-2/scripts/retrain.py " +
#   "--bottleneck_dir={}/bottlenecks ".format(base_path) +
#   "--how_many_training_steps={} ".format(steps) +
#   "--model_dir={}/models/ ".format(base_path) +
#   "--summaries_dir={}/training_summaries/{} ".format(base_path, architecture) +
#   "--output_graph={}/retrained_graph.pb ".format(base_path) +
#   "--output_labels={}/retrained_labels.txt ".format(base_path) +
#   "--architecture={} ".format(architecture) +
#   "--image_dir={}/images ".format(base_path))

cmd = ("python3 vendor/hub/examples/image_retraining/retrain.py " +
    "--image_dir {}/images ".format(base_path) +
    "--tfhub_module https://tfhub.dev/google/imagenet/{}/feature_vector/1 ".format(architecture) +
    "--output_graph={}/retrained_graph.pb ".format(base_path) +
    "--output_labels={}/retrained_labels.txt ".format(base_path) +
    "--how_many_training_steps={} ".format(steps) +
    "--bottleneck_dir={}/bottlenecks ".format(base_path) +
    "--summaries_dir={}/training_summaries/{} ".format(base_path, architecture)
    )
p = subprocess.call(cmd, shell=True)

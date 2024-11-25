import argparse
import datasets
from models.model import *
import yaml


# settings
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, metavar='N', help='the yaml configuration file path')
args = parser.parse_args()
# Add ckpt and checkpoint dir

try:
    with open(args.yaml_conf, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
        yaml_conf["ckpt"] = './results/checkpoints/checkpoints_' + yaml_conf["experiment_name"]

except Exception as e:
    print("Yaml configuration file not successfully loaded:", e)


def train():
    # Define dataloaders
    dataloaders = datasets.get_loaders(yaml_conf)
    # Define model specifications
    if "DSL_mode" in yaml_conf and yaml_conf["DSL_mode"] == "baseline":
        m = Model(yaml_conf=yaml_conf, dataloaders=dataloaders)
    else:
        raise NotImplementedError("DSL mode not implemented")
    # Train the models
    m.train_models()

if __name__ == '__main__':
    # continue to train the model even though some errors happened (dataloader error)
    while True:
        train()
        try:
            train()
        except Exception as e:
            print(f"Error {str(e)} happened! Resume the training process automatically.")
            continue

    


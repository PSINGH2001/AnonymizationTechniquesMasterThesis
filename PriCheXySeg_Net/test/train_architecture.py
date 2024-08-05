import os
import json
import argparse
from utils import utils
import shutil
from agents.Agent import Agent


if __name__ == "__main__":
    print('------------------------------------')
    print('---- Train Anonymization Model -----')
    print('------------------------------------' + '\n')

    # Define an argument parser
    parser = argparse.ArgumentParser('Train Anonymization Model')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_anonymization.json')
    parser.add_argument('--images_path', default='/images/')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config + '\n--images_path: ' + args.images_path + '\n')

    # Read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # Parse config
    config = json.loads(config)
    config['image_path'] = args.images_path

    # Create folder to save experiment-related files
    dir = './archive/' + config['experiment_description']
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    # os.mkdir('./archive/' + config['experiment_description'])
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
    #utils.make_zip(SAVINGS_PATH + config['experiment_description'] + '.zip', './', args.config)

    os.chdir('/home/woody/iwi5/iwi5155h/ExperimentClassifierFreeze/Experiment_0.01/test/')

    # Call agent and run experiment
    experiment = Agent(config)
    experiment.run()

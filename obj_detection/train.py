import os
import sys
from pathlib import Path
import json

def get_training_config(args):
    try:
        if not Path('configs/' + args[1] + '.json').exists():
            print('\n' + args[1] + '.json does not exist')
            for f in list(Path('configs').glob('*.json')):
                print('\t>', f.stem)
            print()
            exit()

    except IndexError:
        print('\nMissing argument: training config file')
        for f in list(Path('configs').glob('*.json')):
            print('\t>', f.stem)
        print()
        exit()

    with open('configs/{}.json'.format(args[1]), 'r') as f:
        conf = json.load(f)

    return conf

def check_pipeline_config(conf):

    config_OK = True
    print('*******************************************************')
    print('Checking config file:', end='\n\n')
    try:
        with open(conf['pipeline_config'], 'rb') as f:
            config_str = str(f.read())
            if not conf['label_map'] in config_str:
                print('Config error: edit label_map path')
                print(conf['label_map'])
                print()
                config_OK = False
            if not conf['checkpoint'] in config_str:
                print('Config error: edit checkpoint path')
                print(conf['checkpoint'])
                print()
                config_OK = False
            if not conf['training_record'] in config_str:
                print('Config error: edit train.tfrecord path')
                print(conf['training_record'])
                print()
                config_OK = False
            # if not str(evaluation_record) in config_str:
            #    print('Config error: edit test.tfrecord path')
            #    print(evaluation_record)
            #    print()
            #    config_OK = False
            if not 'shuffle_buffer_size: ' + str(conf['num_train_examples']) in config_str:
                print('Config warning: Wrong shuffle_buffer size?', conf['num_train_examples'], 'intended')
            if not 'num_steps: ' + str(conf['training_steps']) in config_str:
                print('Config warning: Wrong number of training steps?', conf['training_steps'], 'planed')
            if not 'batch_size: ' + str(conf['batch_size']) in config_str:
                print('Config warning: Wrong batch size?', conf['batch_size'], 'planned')
            print()
    except FileNotFoundError:
        print('pipeline.config not found')
        config_OK = False

    if config_OK:
        print(Path(conf['pipeline_config']).name, 'PASSED simple check...')
    else:
        print(conf['pipeline_config'], 'FAILED simple check...')
    print('*******************************************************\n\n')
    return config_OK

def train_command(conf):
    cmd = 'python {}/model_main.py '.format(    conf['tf_obj_det_dir'])
    cmd+= '--pipeline_config_path={} '.format(  conf['pipeline_config'])
    cmd+= '--model_dir={} '.format(             conf['model_dir'])
    cmd+= '--num_train_steps={} '.format(       conf['training_steps'])
    cmd+= '--num_eval_steps={} '.format(        conf['eval_steps'])
    cmd+= '--alsologtostderr'
    return cmd

def deprecated_train_command(conf):
    cmd = 'python {}/train.py '.format(conf['tf_obj_det_dir'])
    cmd+= '--train_dir={} '.format(conf['training_dir'])
    cmd+= '--pipeline_config_path={} '.format(conf['pipeline_config'])
    cmd+= '--logtostderr '
    return cmd


def freez_graph_command(conf):
    cmd = 'python {}/export_inference_graph.py '.format(conf['tf_obj_det_dir'])
    cmd += '--input_type image_tensor '
    cmd += '--pipeline_config_path {} '.format(conf['pipeline_config'])
    cmd += '--trained_checkpoint_prefix {}/model.ckpt-{} '.format(conf['training_dir'],
                                                                  conf['training_steps'])
    cmd += '--output_directory {}'.format(conf['finetuned_model_dir'])
    return cmd


config = get_training_config(sys.argv)

if not check_pipeline_config(config):
    quit()

### RUN TRAINING FROM CHECKPOINT
if sys.argv[-1] == 'train':

    print(train_command(config), '\n\n')
    os.system(train_command(config))

if sys.argv[-1] == 'old_train':

    print(deprecated_train_command(config), '\n\n')
    os.system(deprecated_train_command(config))


### FREEZING GRAPH'''
if sys.argv[-1] == 'freeze':
    os.system(freez_graph_command(config))


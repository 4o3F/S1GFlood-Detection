import argparse as ag
import json

def parser_with_args(metadata_json='metadata_file.json'):
    parser = ag.ArgumentParser(description='Training flood detection network with S1GFloods')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        

    parser.add_argument('--backbone', default='vitae', type=str, help='type of backbone')
    parser.add_argument('--dataset', default='S1GFloods', type=str, help='type of flood dataset')
    parser.add_argument('--mode', default='rsp', type=str, help='type of pretrained model')
    parser.add_argument('--path', default=None, type=str, help='path of saved best model')

    return parser, metadata
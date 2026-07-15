import argparse as ag
import json


MAX_EPOCHS = 1000


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise ag.ArgumentTypeError('value must be a positive integer')
    return parsed


def epoch_count(value):
    parsed = positive_int(value)
    if parsed > MAX_EPOCHS:
        raise ag.ArgumentTypeError(f'epochs must not exceed {MAX_EPOCHS}')
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise ag.ArgumentTypeError('value must be non-negative')
    return parsed


def parser_with_args(metadata_json='metadata_file.json'):
    parser = ag.ArgumentParser(description='Training flood detection network with S1GFloods')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)

    parser.add_argument('--backbone', default='vitae', type=str, help='type of backbone')
    parser.add_argument('--dataset', default='S1GFloods', type=str, help='type of flood dataset')
    parser.add_argument('--mode', default='rsp', type=str, help='type of pretrained model')
    parser.add_argument('--path', default=None, type=str, help='path of saved best model')
    parser.add_argument('--dataset-dir', default=metadata['dataset_dir'], type=str,
                        help='root directory containing train, val, and test splits')
    parser.add_argument('--epochs', default=metadata['epochs'], type=epoch_count,
                        help=f'maximum training epochs (1-{MAX_EPOCHS})')
    parser.add_argument('--batch-size', default=metadata['batch_size'], type=positive_int)
    parser.add_argument('--loss-function', default=metadata['loss_function'],
                        choices=('hybrid', 'dice', 'bce'))
    parser.add_argument('--validation-interval', default=metadata['validation_interval'],
                        type=positive_int, help='validate every N completed epochs')
    parser.add_argument('--early-stopping-patience',
                        default=metadata['early_stopping_patience'], type=positive_int,
                        help='validation checks allowed without significant F1 improvement')
    parser.add_argument('--min-f1-improvement', default=metadata['min_f1_improvement'],
                        type=nonnegative_float,
                        help='minimum absolute F1 increase required to reset patience')

    return parser, metadata

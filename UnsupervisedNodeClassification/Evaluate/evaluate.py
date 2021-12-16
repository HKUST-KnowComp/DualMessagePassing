import argparse
import numpy as np
from link_prediction import *
from node_classification import *

data_folder, model_folder = '../Data', '../Model'
emb_file, record_file = 'emb.dat', 'record.dat'
link_test_file, label_test_file, label_file = 'link.dat.test', 'label.dat.test', 'label.dat'


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True, type=str, choices=['DBLP', 'Freebase', 'PubMed', 'Yelp'],
        help='Targeting dataset.'
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='Targeting model.',
        choices=[
            'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'R-GIN', 'CompGCN', 'DMPNN', 'HAN',
            'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', 'ConvE'
        ]
    )
    parser.add_argument('--task', required=True, type=str, help='Targeting task.', choices=['nc', 'lp', 'both'])
    parser.add_argument(
        '--attributed',
        required=True,
        type=str,
        help='Only R-GCN, R-GIN, CompGCN, DMPNN, HAN, MAGNN, and HGT support attributed training.',
        choices=['True', 'False']
    )
    parser.add_argument(
        '--supervised',
        required=True,
        type=str,
        help='Only R-GCN, R-GIN, CompGCN, DMPNN, HAN, MAGNN, and HGT support semi-supervised training.',
        choices=['True', 'False']
    )
    parser.add_argument('--emb_file', type=str, default='')
    parser.add_argument('--record_file', type=str, default='')
    parser.add_argument('--link_test_file', type=str, default='')
    parser.add_argument('--label_test_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')

    return parser.parse_args()


def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path, 'r') as emb_file:
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)

    return train_para, emb_dict


def record(args, all_tasks, train_para, all_scores):

    if args.record_file != '':
        filename = args.record_file
    else:
        filename = f'{data_folder}/{args.dataset}/{record_file}'
    with open(filename, 'a') as file:
        for task, score in zip(all_tasks, all_scores):
            file.write(f'model={args.model}, task={task}, attributed={args.attributed}, supervised={args.supervised}\n')
            file.write(f'{train_para}\n')
            if task == 'nc':
                file.write(f'Macro-F1={score[0]:.4f}, Micro-F1={score[1]:.4f}\n')
            elif task == 'lp':
                file.write(f'AUC={score[0]:.4f}, MRR={score[1]:.4f}\n')
            file.write('\n')

    return


def check(args):

    if args.attributed == 'True':
        if args.model not in ['R-GCN', 'R-GIN', 'CompGCN', 'DMPNN', 'HAN', 'MAGNN', 'HGT']:
            print(f'{args.model} does not support attributed training!')
            print('Only R-GCN, R-GIN, CompGCN, DMPNN, HAN, MAGNN, and HGT support attributed training!')
            return False
        if args.dataset not in ['DBLP', 'PubMed']:
            print(f'{args.dataset} does not support attributed training!')
            print('Only DBLP and PubMed support attributed training!')
            return False

    if args.supervised == 'True':
        if args.model not in ['R-GCN', 'R-GIN', 'CompGCN', 'DMPNN', 'HAN', 'MAGNN', 'HGT']:
            print(f'{args.model} does not support semi-supervised training!')
            print('Only R-GCN, R-GIN, CompGCN, DMPNN, HAN, MAGNN, and HGT support semi-supervised training!')
            return False

    return True


def main():

    args = parse_args()

    if not check(args):
        return

    print('Load Embeddings!')
    print(args)
    if args.emb_file != '':
        emb_file_path = args.emb_file
    else:
        emb_file_path = f'{model_folder}/{args.model}/data/{args.dataset}/{emb_file}'
    train_para, emb_dict = load(emb_file_path)

    if args.label_file != '':
        label_file_path = args.label_file
    else:
        label_file_path = f'{data_folder}/{args.dataset}/{label_file}'

    if args.label_test_file != '':
        label_test_file_path = args.label_test_file
    else:
        label_test_file_path = f'{data_folder}/{args.dataset}/{label_test_file}'

    if args.link_test_file != '':
        link_test_file_path = args.link_test_file
    else:
        link_test_file_path = f'{data_folder}/{args.dataset}/{link_test_file}'

    print('Start Evaluation!')
    all_tasks, all_scores = [], []
    if args.task == 'nc' or args.task == 'both':
        print(f'Evaluate Node Classification Performance for Model {args.model} on Dataset {args.dataset}!')
        scores = nc_evaluate(args.dataset, args.supervised, label_file_path, label_test_file_path, emb_dict)
        all_tasks.append('nc')
        all_scores.append(scores)
        print(f'Macro-F1={scores[0]:.4f}, Micro-F1={scores[1]:.4f}')
    if args.task == 'lp' or args.task == 'both':
        print(f'Evaluate Link Prediction Performance for Model {args.model} on Dataset {args.dataset}!')
        scores = lp_evaluate(link_test_file_path, emb_dict)
        all_tasks.append('lp')
        all_scores.append(scores)
        print(f'AUC={scores[0]:.4f}, MRR={scores[1]:.4f}')

    print('Record Results!')
    record(args, all_tasks, train_para, all_scores)

    return


if __name__ == '__main__':
    main()

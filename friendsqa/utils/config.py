import os
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='Parameters for FriendsQA dataset')

parser.add_argument('-lr', '--learning_rate', type=float, default=4e-6)
parser.add_argument('-cd', '--cuda', type=int, default=0)
parser.add_argument('-sd', '--seed', type=int, default=1919810)
parser.add_argument('-eps', '--epochs', type=int, default=3)
parser.add_argument('-mgr', '--max_grad_norm', type=float, default=1.0)
parser.add_argument('-dp', '--data_path', type=str, default='data')
parser.add_argument('-mt', '--model_type', type=str, default='electra')
parser.add_argument('-cp', '--cache_path', type=str, default='cache')
parser.add_argument('-ml', '--max_length', type=int, default=512)
parser.add_argument('-qml', '--question_max_length', type=int, default=32)
parser.add_argument('-bsz', '--batch_size', type=int, default=4)
parser.add_argument('-elsp', '--early_stop_patience', type=int, default=10)
parser.add_argument('-dbg', '--debug', type=bool, default=False)
parser.add_argument('-wmprop', '--warmup_proportion', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-6)
parser.add_argument('--small', type=bool, default=False, help='whether to use small dataset')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--colab', type=bool, default=False)
parser.add_argument('--pred_file', type=str, default='pred.json')
parser.add_argument('--mha_layer_num', type=int, default=3)
parser.add_argument('--model_num', type=int, default=1)
parser.add_argument('--use_cls_for_gather', type=int, default=0, help='whether to use [CLS] as the token to gather speaker/utter information')
parser.add_argument('--draw', type=int, default=0, help='whether to draw attention')

args = parser.parse_args()

if not os.path.exists('saves' if not args.small else 'saves_small'):
    os.mkdir('saves' if not args.small else 'saves_small')
if not os.path.exists('caches' if not args.small else 'caches_small'):
    os.mkdir('caches' if not args.small else 'caches_small')

args.add_speaker_mask = args.model_num

args.save_path = ('saves/' if not args.small else 'saves_small/') + args.model_type + '_' + args.save_path
args.cache_path = ('caches/' if not args.small else 'caches_small/') + args.model_type + '_' + args.cache_path
args.model_name = 'google/electra-large-discriminator'

args.cache_path += '_' + str(args.max_length)
args.save_path += '_' + str(args.max_length) + '_' + str(args.seed)
if args.use_cls_for_gather:
    args.cache_path += '_cls'
    args.save_path += '_cls'
args.pred_file = args.save_path + '/' + args.pred_file

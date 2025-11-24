import argparse

parser = argparse.ArgumentParser(description='patch')

# basic settings
parser.add_argument('--txt_enc', type=str, required=False, default='bert')
parser.add_argument('--img_enc', type=str, required=False, default='vit16', choices=['vit16', 'resnet50'])

parser.add_argument('--method', type=str, required=True)
parser.add_argument('--dataset', type=str, default='vireo')
parser.add_argument('--data_path', type=str, default=r'/Vireo172_Image')
parser.add_argument('--bert_path', type=str, default=r'/mnt/cross_modal/bert_pretrain_add_ingredients')
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--pad_size', type=int, default=32, help='bert sequence padding length')
parser.add_argument('--save_path', type=str, default=r'./checkpoints', help='path to save the training log and checkpoints')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--training', type=str, default="True")
parser.add_argument('--lr', nargs=3, type=float, required=False, default=[5e-5, 5e-5, 5e-5], help='learning rate')
parser.add_argument('--lr_decay', nargs=2, type=float, default=[3, 0.5])
parser.add_argument('--seed', type=int, default=42, help='miracle')  # vit-32: 49;vit-16:196
parser.add_argument('--pos_weight', type=int, default=40, help='weight for BCE loss')

# distributed settings
parser.add_argument('--worker', type=int, default=4)
parser.add_argument('--distributed', action="store_true")

# patch settings
parser.add_argument('--pred_text_path', type=str, default='/mnt/patch_data/text_prompt_topk')
parser.add_argument('--patch_path', type=str, default='/mnt/patch_data')
parser.add_argument('--patch_pad_len', type=int, default=10)
parser.add_argument('--text_pad_len', type=int, default=25)
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--use_text', action="store_true")
parser.add_argument('--patch_level_info_k', type=int, default=2048)
parser.add_argument('--thresh', type=float, default=0.15)
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()
args.training = True if args.training == "True" else False

args.max_words = args.pad_size
if args.dataset == 'vireo':
    args.num_class = 172
    args.f_path = '/Vireo172_Text'
    args.img_path = '/Vireo172_Image'
    args.load_dino_info = False
    args.extra_image_info_path = None if True else '/mnt/patch_data/ingre2image_info/sorted_filtered_word2label2image_info.json'
    args.confounder_path = '/mnt/cross_modal/ingre_confounder_dict.pt'
    args.patch_level_info_path = '/mnt/cross_modal/features/vireo_all_patch_info.pkl'
    args.patch_level_info_path = 'features/patch_info/test_patch_info.pt'
    args.patch_level_info = True
    args.patch_level_info_k = 2048

    # args.thresh = 0.2
    args.patch_pad_len = 10

elif args.dataset == 'food101':
    args.num_class = 101

elif args.dataset == 'wide':
    args.num_class = 81
    args.f_path = '/data_NUS_WIDE'
    args.img_path = '/data_NUS_WIDE/NUS_images'
    args.patch_info_path = 'features/patch_info'
    args.pred_tag_path = '/mnt/patch_data/text_prompt_topk'
import torch
import numpy as np
from datetime import datetime
import os

from models.bert import BertModel, BertTokenizer
from data.data import build_dataset, get_patch
from models.patch_wrapper import PatchWrapper, ConfounderWrapper, CausalModule, PatchLevelConfounderWrapper, DualPathFuseModule, PreprocessedPatchWrapper
from models.vit_backbone import VisionTransformer
from models.patch_gat import PatchGAT
from train_eval.train_patch import train_patch_vit, test_patch_vit, specific_sample_analysis, extract_patch_feat, save_logits, pre_extract_patch_feat
from utils import setup_logger, set_requires_grad
from opts.patch_opt import args
import torch.nn as nn
import ipdb
import random
import torch.distributed as dist


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    args.distributed = 'WORLD_SIZE' in os.environ
    if args.distributed:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.rank)
    else:
        args.local_rank = 0
        args.world_size = 1

    if args.world_size == 1:
        args.distributed = False

    gpu_count = torch.cuda.device_count()
    gpu_names = torch.cuda.get_device_name(0)

    # initial settings
    if args.local_rank == 0:
        time_stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        args.save_path = os.path.join(args.save_path, args.dataset, args.method + ("" if args.training else "-test"),
                                      time_stamp)
        logger = setup_logger(f'{args.method.upper()}', save_dir=args.save_path, if_train=args.training)
        logger.info(f'Training log is saved to {args.save_path}')
        logger.info(f'Training begins at {time_stamp}')
        logger.info(f'Number of GPUs : {gpu_count}')
        logger.info(f'GPU model: {gpu_names}')
        # cuda version is the pytorch compatible version
        logger.info(f'pytorch version: {torch.__version__}, cuda version: {torch.version.cuda}')

    # load data 
    if args.method != 'extract_patch_feat':
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        if args.dataset == 'food101':
            pass
        elif args.dataset == 'vireo':
            train_loader = build_dataset('train', tokenizer, args)
            test_loader = build_dataset('test', tokenizer, args)
        elif args.dataset == 'wide':
            train_loader = build_dataset('train', tokenizer, args)
            test_loader = build_dataset('test', tokenizer, args)
        else:
            raise ValueError(f'Unrecognized dataset: {args.dataset}')

    # # load bert
    # bert = BertModel.from_pretrained(args.bert_path).to(args.device)

    if args.method == 'fusion_patch_vit':
        model = PatchWrapper(args, vit_classifier=True).to(args.device)
        model = nn.DataParallel(model)

        train_patch_vit(args, model, train_loader, test_loader)
    elif args.method == 'patch_vit':
        args.intervention_classifier = True
        args.front_door = False
        model = PatchWrapper(args, vit_classifier=False, intervention_classifier=args.intervention_classifier, front_door=args.front_door).to(args.local_rank)
        # model = PreprocessedPatchWrapper(args, intervention_classifier=args.intervention_classifier).to(args.local_rank)
        state_dict = torch.load(
                                # "/mnt/cross_modal/checkpoints/epoch_9_ckpt_90.45.pt",
                                # front-door
                                # "/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-11-19 00-06-40/best_checkpoint_PATCH_VIT.pth",
                                # new ckpt 
                                "/mnt/cross_modal/checkpoints/vireo/patch_vit/2025-03-21 09-54-14/PATCH_VIT_0.9045723676681519_ckpt.pth",
                                map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        # model.load_state_dict(new_state_dict)

        # state_dict = torch.load(
        #     "/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-11-19 00-06-40/best_checkpoint_PATCH_VIT.pth",
        #     map_location=f'cuda:{args.local_rank}')
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     new_key = k.replace('module.', '') if k.startswith('module.') else k
        #     new_state_dict[new_key] = v
        #
        # model.load_state_dict(new_state_dict, strict=True)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

        # logits mean only patch
        # model.load_state_dict(torch.load("/mnt/patch/checkpoints/vireo/patch_vit/2024-07-08 15-19-58/best_epoch19_acc0.8831211924552917.pth"))
        # ipdb.set_trace()

        # self-attention only patch
        # model.load_state_dict(torch.load("/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-07-11 16-20-09/best_epoch18_acc0.8821258544921875.pth"))
        # model.load_state_dict(torch.load("/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-07-25 20-46-53/best_epoch20_acc0.9053809642791748.pth"))
        # model.load_state_dict(torch.load("/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-07-22 10-42-20/best_epoch15_acc0.8849309086799622.pth"))
        # model.load_state_dict(torch.load("/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-09-03 10-40-07/best_checkpoint_PATCH_VIT.pth"))

        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            save_logits(args, model, test_loader)
            # specific_sample_analysis(args, model.eval(), test_loader, '/2/3_36.jpg')
            test_patch_vit(args, model, test_loader)

    # elif args.method == 'extra_patch':
    #     model = PatchWrapper(args, vit_classifier=False, use_extra_image=True).to(args.local_rank)
    #     if args.distributed:
    #         model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    #     if args.training:
    #         train_patch_vit(args, model, train_loader, test_loader)

    elif args.method == 'confounder_backdoor':
        
        model = ConfounderWrapper(args, vit_classifier=False).to(args.local_rank)
        state_dict = torch.load(
            # "/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-11-19 00-06-40/best_checkpoint_PATCH_VIT.pth",
            # "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/rare_word_patch_avg_feat_as_confounder91.32_ckpt.pth",
            # "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/pcl+uloss_attn_confounder_91.36.pth",
            "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/cluster_confounder_91.34.pth",
            map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # model.load_state_dict(new_state_dict, strict=True)

        # set_requires_grad(model, False)
        # set_requires_grad(model.rare_fusion, True)
        # set_requires_grad(model.logit_clf, True)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            test_patch_vit(args, model, test_loader)

    elif args.method == 'fuse2method':
        model = CausalModule(args).to(args.local_rank)
        state_dict = torch.load(
            # wide
            # "checkpoints/wide/patch_vit/PATCH_VIT_best_recall0.47569252672975426_ckpt.pth",
            # vireo
            "/mnt/cross_modal/checkpoints/vireo/patch_vit/2024-11-19 00-06-40/best_checkpoint_PATCH_VIT.pth",
            # "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/rare_word_patch_avg_feat_as_confounder91.32_ckpt.pth",
            # "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/pcl+uloss_attn_confounder_91.36.pth",
            map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        frontdoor_classifier_weight = new_state_dict['wrapper_classifier.weight']
        model.front_door_module.load_state_dict(new_state_dict, strict=True)
        # set_requires_grad(model.front_door_module, False)

        state_dict = torch.load(
            # "/public/localUsers/wuwei/cross_modal/checkpoints/epoch_9_ckpt_90.45.pt",
            # "/mnt/cross_modal/checkpoints/vireo/confounder_backdoor/rare_word_patch_avg_feat_as_confounder91.32_ckpt.pth",
            # vireo
            "/mnt/cross_modal/checkpoints/vireo/patch_level_confounder/2024-12-04 22-25-18/PATCH_LEVEL_CONFOUNDER_0.9131077527999878_ckpt.pth",
            # vireo with text
            # "checkpoints/vireo/patch_level_confounder/2024-12-18 16-32-37/PATCH_LEVEL_CONFOUNDER_0.9130474328994751_ckpt.pth",
            # wide
            # "checkpoints/wide/patch_level_confounder/PATCH_LEVEL_CONFOUNDER_best_recall0.4750813545655916_ckpt.pth",
            # wide with text
            # "checkpoints/wide/patch_level_confounder/2025-01-10 10-50-08/PATCH_LEVEL_CONFOUNDER_best_recall0.4744033497249011_ckpt.pth",
            map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        patch_confounder_classifier_weight = new_state_dict['logit_clf.weight']
        model.patch_level_confounder.load_state_dict(new_state_dict, strict=True)
        # set_requires_grad(model.patch_level_confounder, False)

        state_dict = torch.load(
            "/mnt/cross_modal/checkpoint/vit16_baseline_acc88.51.pth",
            # "checkpoints/wide/base_vit/best_epoch0_recall0.45036091488392066.pth",
            map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        base_classifier_weight = new_state_dict['classifier.weight']
        model.base.load_state_dict(new_state_dict, strict=False)

        # state_dict = torch.load(
        #     # vireo
        #     # "/mnt/cross_modal/checkpoint/epoch_9_ckpt_90.45.pt",
        #     # wide
        #     "checkpoints/wide/patch_vit/PATCH_VIT_best_recall0.4637161733158_ckpt.pth",
        #     map_location=f'cuda:{args.local_rank}')
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     new_key = k.replace('module.', '') if k.startswith('module.') else k
        #     new_state_dict[new_key] = v
        # patch_relation_classifier_weight = new_state_dict['wrapper_classifier.weight']
        # model.patch_relation_modeling.load_state_dict(new_state_dict, strict=True)

        classifier_weight = torch.cat((frontdoor_classifier_weight, patch_confounder_classifier_weight, base_classifier_weight), dim=1)
        model.classifier.weight = nn.Parameter(classifier_weight, requires_grad=False)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            save_logits(args, model, test_loader)
            test_patch_vit(args, model, test_loader)

    elif args.method == 'patch_level_confounder':
        # ipdb.set_trace()
        model = PatchLevelConfounderWrapper(args, use_text=args.use_text).to(args.local_rank)
        # 91.22 acc1
        state_dict = torch.load("/mnt/cross_modal/checkpoints/vireo/patch_level_confounder/2024-12-04 22-25-18/PATCH_LEVEL_CONFOUNDER_0.9131077527999878_ckpt.pth", map_location=f'cuda:{args.local_rank}')
        # w/o loss constraints -> acc1: 90.71
        # state_dict = torch.load("checkpoints/vireo/patch_level_confounder/2025-04-16 10-43-36/PATCH_LEVEL_CONFOUNDER_0.9071058630943298_ckpt.pth")
        # state_dict = torch.load("checkpoints/wide/base_vit/2024-12-23 09-45-51/best_epoch0_recall0.45036091488392066.pth")
        # state_dict = torch.load("checkpoints/wide/base_vit/2024-12-24 19-42-37/best_epoch4_recall0.4771975379409617.pth")
        # state_dict = torch.load("checkpoints/vireo/patch_level_confounder/2024-12-18 16-32-37/PATCH_LEVEL_CONFOUNDER_0.9130474328994751_ckpt.pth")
        # state_dict = torch.load("checkpoints/wide/patch_level_confounder/PATCH_LEVEL_CONFOUNDER_best_recall0.4750813545655916_ckpt.pth")
        # state_dict = torch.load("checkpoints/wide/patch_level_confounder/2025-01-19 18-07-00/PATCH_LEVEL_CONFOUNDER_best_recall0.4722588219308432_ckpt.pth")
        # # ipdb.set_trace()
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        # del new_state_dict['classifier.weight'], new_state_dict['classifier.bias']
        model.load_state_dict(new_state_dict, strict=False)
        param_dict = dict(model.named_parameters())
        # ipdb.set_trace()
        # frozen to train text embedding
        for k in new_state_dict.keys():
            # if k.startswith('transformer.') or k.startswith('embedding.') or k.startswith('cls_token'):
            if not k == 'logit_clf' and not k.startswith('aggr') and not k.startswith('txt') :
                p = param_dict[k]
                if p is None:
                    ipdb.set_trace()
                p.requires_grad = False

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            save_logits(args, model, test_loader)
            # test_patch_vit(args, model, test_loader)

    elif args.method == "extract_patch_feat":
        # loader = get_patch(args, root="/mnt/patch_data/wide/TRAIN_REGIONS_BY_PREDICTING_TOP5/")
        vireo_train_loader = get_patch(args, root='/mnt/patch_data/TEST_REGIONS_BY_PREDICTING_TOP3_THRESH/')
        model = VisionTransformer(args, use_classifier=False).to(f'cuda:{args.local_rank}')
        # state_dict = torch.load("checkpoint/best_checkpoint_PATCH_VIT90.84.pth", map_location=f'cuda:{args.local_rank}')
        state_dict = torch.load("checkpoint/vit16_baseline_acc88.51.pth", map_location=f'cuda:{args.local_rank}')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        
        # state_dict = torch.load(
        #     "checkpoints/wide/patch_vit/2024-12-22 13-35-50/PATCH_VIT_best_recall0.47663273887550145_ckpt.pth",
        #     map_location=f'cuda:{args.local_rank}')
        model.load_state_dict(new_state_dict, strict=False)
        
        extract_patch_feat(args, vireo_train_loader, model)

    elif args.method == 'fuse_text':
        model = DualPathFuseModule(args).to(args.local_rank)
        state_dict = torch.load(
            "checkpoints/wide/base_vit/2024-12-24 19-42-37/best_epoch4_recall0.4771975379409617.pth")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            test_patch_vit(args, model, test_loader)

    elif args.method == 'pre_extract_patch_feat':
        model = VisionTransformer(args, use_classifier=False).to(f'cuda:{args.local_rank}')
        state_dict = torch.load(
            "checkpoint/vit16_baseline_acc88.51.pth", map_location=f'cuda:{args.local_rank}')
        model.load_state_dict(state_dict, strict=False)
        pre_extract_patch_feat(args, test_loader, model)
        
    elif args.method == 'patch_gat':
        args.interventional_classifier = True
        model = PatchGAT(args, interventional_classifier=args.interventional_classifier).to(args.local_rank)
        
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        if args.training:
            train_patch_vit(args, model, train_loader, test_loader)
        else:
            save_logits(args, model, test_loader)

        
    else:
        raise ValueError('please refer to other main.py file')

import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score, f1_score
from scipy.ndimage import gaussian_filter
from dataset.zero import TestDataset, TrainDataset
import CLIP
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from CLIP.moe_adapter import CLIP_MoE_Inplanted
from torchvision import transforms
import cv2
import re
import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Fire_exits': 3, 'Fire_exits': 2, 'Fire_exits': 1, 'Fire_exits': -1, 'Fire_exits': -2, 'Fire_exits': -3,
               'Fire_exits': 4}
CLASS_INDEX_INV = {3: 'Fire_exits', 2: 'Fire_exits', 1: 'Fire_exits', -1: 'Fire_exits', -2: 'Fire_exits', -3: 'Fire_exits',
                   4: 'Fire_exits'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).float().sum()
    union = torch.logical_or(mask1, mask2).float().sum()
    iou = intersection / union
    return iou.item()

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--obj', type=str, default='Fire_exits')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument('--save_path_moe_adapter', type=str, default='./ckpt/fire/64_8')
    parser.add_argument("--epoch", type=int, default=500, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)

    # LoRA arguments
    parser.add_argument('--r', default=64, type=int)
    parser.add_argument('--alpha', default=8, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'])
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', help='a list of strings', default=['q', 'v', 'o'])
    parser.add_argument('--save_path_lora', default='./ckpt/fire/64_8', help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')

    args = parser.parse_args()

    setup_seed(args.seed)

    #import pdb;
    #pdb.set_trace()

    # fixed feature extractor
    clip_model, preprocess = CLIP.load(args.model_name)
    clip_model.eval()
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    #model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model = CLIP_MoE_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    #load parameter
    checkpoint = torch.load(os.path.join(f'{args.save_path_moe_adapter}', f'{args.obj}.pth'))
    model.seg_MoE_adapters.load_state_dict(checkpoint["seg_MoE_adapters"])
    model.det_MoE_adapters.load_state_dict(checkpoint["det_MoE_adapters"])
    load_lora(args, list_lora_layers)

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = TrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = TestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [3,2,1,-1,-2,-3,4]:
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])


def test(args, seg_model, test_loader, text_features):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []

    for (image, y, mask, path) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        #print(path[0])

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            # image
            anomaly_score = 0
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]

                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)

            #visualization
            # filename = re.split(r'[/\\]', path[0])[-1]
            # img_size = 336
            # ori_img = cv2.imread(path[0])
            # ori_img = cv2.resize(ori_img, (img_size, img_size))
            # vis = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # RGB
            # mask_vi = normalize(final_score_map[0])
            # vis = apply_ad_scoremap(vis, mask_vi)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # save_path = 'result/zero_shot/r64_a128/test'
            # save_vis = os.path.join(save_path)
            # if not os.path.exists(save_vis):
            #     os.makedirs(save_vis)
            # cv2.imwrite(os.path.join(save_vis, filename), vis)


            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    gt_mask_list = gt_mask_list.flatten()

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())
    segment_scores = segment_scores.flatten()

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    image_scores[image_scores > 0.5], image_scores[image_scores <= 0.5] = 1, 0
    img_accuracy_det = accuracy_score(gt_list, image_scores)
    img_precision_det = precision_score(gt_list, image_scores)
    img_recall_det = recall_score(gt_list, image_scores)
    img_f1_det = f1_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det, 6)}')
    print(f'accuracy : {round(img_accuracy_det, 6)}\n')
    print(f'precision : {round(img_precision_det, 6)}\n')
    print(f'recall : {round(img_recall_det, 6)}\n')
    print(f'f1 : {round(img_f1_det, 6)}\n')
    all = img_roc_auc_det + img_accuracy_det + img_precision_det + img_recall_det + img_f1_det

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list, segment_scores)
        segment_scores[segment_scores > 0.5], segment_scores[segment_scores <= 0.5] = 1, 0
        seg_accuracy = accuracy_score(gt_mask_list, segment_scores)
        seg_precision = precision_score(gt_mask_list, segment_scores)
        seg_recall = recall_score(gt_mask_list, segment_scores)
        seg_f1 = f1_score(gt_mask_list, segment_scores)
        gt_mask_list = torch.from_numpy(gt_mask_list)
        segment_scores = torch.from_numpy(segment_scores)
        iou = calculate_iou(gt_mask_list, segment_scores)
        all = all + seg_roc_auc + seg_accuracy + seg_precision + seg_recall + seg_f1 + iou
        all = all / 11
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 6)}')
        print(f'seg_accuracy : {round(seg_accuracy, 6)}\n')
        print(f'seg_precision : {round(seg_precision, 6)}\n')
        print(f'seg_recall : {round(seg_recall, 6)}\n')
        print(f'seg_f1 : {round(seg_f1, 6)}\n')
        print(f'iou : {round(iou, 6)}\n')
        print(f'All-Average : {round(all, 6)}\n')
        return all
    else:
        return img_roc_auc_det


if __name__ == '__main__':
    main()



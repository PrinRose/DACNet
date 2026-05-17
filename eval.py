import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import metrics


def Borders_Capture(gt, pred, dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(gt)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)
    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))
    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge, mergep, np.sum(img_dilate) / 255


def _eval_one_split(dataset, split_name, gt_dir, pred_dir, args):

    if not os.path.exists(pred_dir):
        return

    name_list = sorted(os.listdir(pred_dir))
    if len(name_list) == 0:
        return

    FM = metrics.Fmeasure_and_FNR()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    MAE = metrics.MAE()

    BR_MAE = metrics.MAE()
    BR_wF = metrics.WeightedFmeasure()

    for name in tqdm(name_list, desc=f"Evaluating {dataset} [{split_name}]"):
        base = os.path.splitext(name)[0]
        gt_path = os.path.join(gt_dir, base + '.png')
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(gt_path):
            continue

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt is None or pred is None:
            continue

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            cv2.imwrite(pred_path, pred)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

        if args.BR == 'on':
            BR_gt, BR_pred, area = Borders_Capture(
                cv2.imread(gt_path), cv2.imread(pred_path), int(args.br_rate))
            BR_MAE.step(pred=BR_pred, gt=BR_gt, area=area)
            BR_wF.step(pred=BR_pred, gt=BR_gt)

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]

    def r(x): return '-' if x is None else str(x.round(3))

    lines = [
        f"Model:{args.model}, Dataset:{dataset}, Split:{split_name} ||",
        f"Smeasure:{r(sm)}; meanEm:{r(em['curve'].mean() if em['curve'] is not None else None)}; "
        f"wFmeasure:{r(wfm)}; MAE:{r(mae)}; fnr:{r(fnr)}; ",
        f"adpEm:{r(em['adp'])}; maxEm:{r(em['curve'].max() if em['curve'] is not None else None)}; ",
        f"adpFm:{r(fm['adp'])}; meanFm:{r(fm['curve'].mean())}; maxFm:{r(fm['curve'].max())}"
    ]

    if args.BR == 'on':
        br_mae = BR_MAE.get_results()['mae']
        br_wf = BR_wF.get_results()['wfm']
        lines.append(f"BR{args.br_rate}_mae:{r(br_mae)}; BR{args.br_rate}_wF:{r(br_wf)}")

    result_str = ' '.join(lines)
    print(result_str)
    print("#" * 60)

    if args.record_path:
        with open(args.record_path, 'a') as f:
            f.write(result_str + '\n')


def eval_dataset(dataset, args):
    gt_dir = os.path.join(args.GT_root, dataset, 'GT')
    base_pred_dir = os.path.join(args.pred_root, 'pred_masks', dataset)

    _eval_one_split(dataset, 'all', gt_dir, base_pred_dir, args)

    for split_name in ['small', 'medium', 'large']:
        split_pred_dir = os.path.join(base_pred_dir, split_name)
        _eval_one_split(dataset, split_name, gt_dir, split_pred_dir, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='COD')
    parser.add_argument("--pred_root", default='../output')
    parser.add_argument("--GT_root", default='../COD-TestDataset')
    parser.add_argument("--record_path", default='../eval_record.txt')
    parser.add_argument("--BR", default='off')
    parser.add_argument("--br_rate", type=int, default=15)
    args = parser.parse_args()

    all_datasets = ['CAMO', 'COD10K', 'NC4K']
    for dataset in all_datasets:
        pred_dir = os.path.join(args.pred_root, 'pred_masks', dataset)
        if os.path.exists(pred_dir):
            eval_dataset(dataset, args)
        else:
            print(f"跳过 {dataset}: 未找到预测目录 {pred_dir}")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import json
import logging
from glob import glob
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np



# ----------------------------
# Configurable Parameters
# ----------------------------
# gate_controller.py から設定値をインポート
from config import (
    PARENT_DIR, IMG_DIR, DATASET_DIR, JSON_DIR, OUT_DIR, IMG_OUT_DIR, INPUT_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EVAL_PERIOD, NUM_WORKERS, DIST_THRESH, PRED_CKPT,
    HEATMAP_CMAP, HEATMAP_IMG_CMAP, FEATUREMAP_CMAP, POINT_ERROR_COLORS,POINT_LABEL,HEATIMG_OUT_DIR
)
input_size=INPUT_SIZE[0]


#? jsonアノテーションファイルをtrain/val/testに分割する
# 入力: json_dir(str), ratios(tuple[float,float,float]), seed(int)
# 出力: (train_jsons(list[str]), val_jsons(list[str]), test_jsons(list[str]))
def split_dataset(json_dir, ratios=(0.7,0.15,0.15), seed=42):#seedを固定することで、毎回同じ画像が同じ所に分類される...はず
    paths = glob(os.path.join(json_dir, '*.json'))
    random.seed(seed); random.shuffle(paths)
    n = len(paths)
    n1 = int(n * ratios[0]); n2 = n1 + int(n * ratios[1])
    return paths[:n1], paths[n1:n2], paths[n2:]

#? モデルの平均誤差・全誤差リスト・各点ごとの誤差リストを計算する
# 入力: model(torch.nn.Module), loader(torch.utils.data.DataLoader), device(str|torch.device)
# 出力: (mean_error(float), errors(list[float]), point_errors(dict[str, list[float]]))
def mean_error(model, loader, device):
    global input_size
    model.eval()
    errors = []
    point_labels = POINT_LABEL
    point_errors = {lbl: [] for lbl in point_labels}
    with torch.no_grad():
        for imgs, targets, masks, gate_exists in loader:#!エラーの平均値にゲート存在確率は含まない
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            out = model(imgs)  # [B,9]
            preds = out[:, :8].cpu().numpy()  # [B,8]
            tars  = targets.cpu().numpy()
            ms    = masks.cpu().numpy()
            for p, t, m in zip(preds, tars, ms):
                for i, lbl in zip(range(0, len(point_labels)*2, 2), point_labels):
                    if m[i] == 0: continue
                    gt = np.array([t[i], t[i+1]])
                    pr = np.array([p[i], p[i+1]])
                    err = np.linalg.norm(gt - pr)
                    err *=input_size
                    errors.append(err)
                    point_errors[lbl].append(err)
    return (np.mean(errors) if errors else 0.0, errors, point_errors)

#? 誤差リストから最大値を返す
# 入力: errors(list[float])
# 出力: 最大誤差(float)
def max_error(errors):
    return np.max(errors) if errors else 0.0

#? 指定閾値以下の誤差の割合（正解率）を返す
# 入力: errors(list[float]), threshold(float)
# 出力: accuracy(float)
def accuracy_at_threshold(errors, threshold):
    errors = np.array(errors)
    return np.mean(errors < threshold) if len(errors) > 0 else 0.0


#? 各点ごとの平均誤差をヒートマップ画像として保存する
# 入力: model(torch.nn.Module), loader(torch.utils.data.DataLoader), device(str|torch.device), session_dir(str)
# 出力: 画像ファイル(str)
def plot_heatmap(model, loader, device, session_dir):
    global input_size
    import seaborn as sns
    model.eval()
    point_labels = POINT_LABEL
    point_errors = {lbl: [] for lbl in point_labels}
    with torch.no_grad():
        for imgs, targets, masks, gate_exists in loader:
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
            out = model(imgs)
            preds = out[:, :8].cpu().numpy()
            tars  = targets.cpu().numpy()
            ms    = masks.cpu().numpy()
            for p, t, m in zip(preds, tars, ms):
                for i, lbl in zip(range(0, len(point_labels)*2, 2), point_labels):
                    if m[i] == 0: continue
                    gt = np.array([t[i], t[i+1]])
                    pr = np.array([p[i], p[i+1]])
                    err = np.linalg.norm(gt - pr)
                    err *=input_size
                    point_errors[lbl].append(err)
    # ヒートマップ用データ作成
    # 各点ごとの誤差リストの平均値を使う（1行4列の2次元配列にする）
    data = [[np.mean(point_errors[lbl]) if len(point_errors[lbl]) > 0 else 0.0 for lbl in point_labels]]
    plt.figure(figsize=(8, 3))  # 長方形比率
    ax = plt.gca()
    sns.heatmap(data, annot=True, fmt='.1f', cmap=HEATMAP_CMAP, xticklabels=point_labels, yticklabels=['error'], square=True, cbar=True, ax=ax)
    plt.title('Error Heatmap by Point')
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, 'pointwise_error_heatmap.png'), dpi=300)
    plt.close()

#? 画像1枚に対する各点誤差のヒートマップを保存する
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device), out_path(str)
# 出力: 成功可否(bool)
def plot_heatmap_for_image(model, img_path, device, out_path):
    import seaborn as sns
    model.eval()
    point_labels = POINT_LABEL
    # 画像読み込み・前処理
    transform = T.Compose([T.Resize(INPUT_SIZE), T.Grayscale(1), T.ToTensor()])
    orig = Image.open(img_path).convert('RGB')
    x = transform(orig).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pts = out[0, :8].cpu().numpy()
    # 予測点を画像サイズに戻す
    w0, h0 = orig.size
    sx = w0 / INPUT_SIZE[1]
    sy = h0 / INPUT_SIZE[0]
    pts_orig = [(pts[i]*sx, pts[i+1]*sy) for i in range(0, 8, 2)]
    # 正解点（jsonから取得）
    # jsonはjsonフォルダ内
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(JSON_DIR, base + '.json')
    if not os.path.exists(json_path):
        print(f"対応するjsonファイルが見つかりません: {json_path}")
        return False
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_map = {s['label']: s['points'][0] for s in data['shapes'] if s.get('shape_type') == 'point'}
    gt_pts = [gt_map.get(lbl, (0,0)) for lbl in point_labels]
    # 各点の誤差
    errors = [np.linalg.norm(np.array(gt) - np.array(pr)) for gt, pr in zip(gt_pts, pts_orig)]
    # 2次元配列にしてヒートマップ
    arr = np.array(errors).reshape(1, -1)
    plt.figure(figsize=(8, 2))
    ax = plt.gca()
    sns.heatmap(arr, annot=True, fmt='.1f', cmap=HEATMAP_IMG_CMAP, xticklabels=point_labels, yticklabels=['error'], square=True, cbar=True, ax=ax)
    plt.title('Error Heatmap for Image')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True

#? 画像1枚の推論＋特徴マップ保存＋ヒートマップ生成
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device), out_dir(str)
# 出力: (pts_orig(list[tuple[float,float]]), out_img_path(str), heatmap_path(str|None), fmap_dir(str), None)
def predict_with_features(model, img_path, device, out_dir):
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image, ImageDraw
    import torchvision.transforms as T

    model.eval()
    point_labels = POINT_LABEL  # ['1', '2', '3', '4']

    # ── 画像の前処理 ──
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    orig = Image.open(img_path).convert('RGB')
    x = transform(orig).unsqueeze(0).to(device)

    # ── フックをつけて特徴マップを取得 ──
    features = {}
    def hook_fn(module, input, output):
        features['fcnn'] = output.detach().cpu().numpy()
    handle = model.pool4.register_forward_hook(hook_fn)

    # ── 推論 ──
    with torch.no_grad():
        out = model(x)
        # out[0, :8] は「0～1 の正規化座標」が返ってくる前提
        pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)
        gate_prob = torch.sigmoid(out[0, 8]).item()
    handle.remove()

    # ── 元画像サイズを取得 ──
    w0, h0 = orig.size  # (元の幅, 元の高さ)

    # ── normalized → pixel に戻す ──
    pts_orig = []
    for (x_n, y_n) in pts_norm:
        x_px = x_n * w0
        y_px = y_n * h0
        pts_orig.append((x_px, y_px))

    # ── 予測点を描画 ──
    draw_img = orig.copy()
    draw = ImageDraw.Draw(draw_img)
    colors = ['red', 'blue', 'green', 'yellow']
    for (x_, y_), c in zip(pts_orig, colors):
        r = 5
        draw.ellipse((x_-r, y_-r, x_+r, y_+r), fill=c)

    # ── 4点間を結ぶ線を引く ──
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts_orig
    draw.line([(x1, y1), (x2, y2)], fill="white", width=2)
    draw.line([(x1, y1), (x3, y3)], fill="white", width=2)
    draw.line([(x1, y1), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x3, y3)], fill="white", width=2)
    draw.line([(x3, y3), (x4, y4)], fill="white", width=2)

    out_img_path = os.path.join(out_dir, 'predict.jpg')
    draw_img.save(out_img_path)

    # ── GT JSON があれば、誤差を計算してヒートマップなど生成可能 ──
    # base = os.path.splitext(os.path.basename(img_path))[0]
    # json_path = os.path.join(JSON_DIR, base + '.json')
    # heatmap_path = os.path.join(out_dir, 'heatmap.jpg')
    # if os.path.exists(json_path):
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #     gt_map = {s['label']: s['points'][0] for s in data['shapes'] if s.get('shape_type') == 'point'}
    #     gt_pts = [tuple(gt_map.get(lbl, (0, 0))) for lbl in point_labels]
    #     errors = [np.linalg.norm(np.array(gt) - np.array(pr)) for gt, pr in zip(gt_pts, pts_orig)]

    #     plt.figure(figsize=(8, 2))
    #     sns.heatmap([errors], annot=True, fmt=".1f", cmap='Reds')
    #     plt.title("Point-wise Error (px)")
    #     plt.xlabel("Point Index")
    #     plt.ylabel("Error")
    #     plt.savefig(heatmap_path)
    #     plt.close()
    # else:
    #     heatmap_path = None
    heatmap_path = None

    # ── 特徴マップを保存するディレクトリを作成 ──
    fmap_dir = os.path.join(out_dir, 'features_map')
    save_all_fmap(model, x, fmap_dir)

    return pts_orig, out_img_path, heatmap_path, fmap_dir, gate_prob



#? モデルの全特徴マップを画像として保存する
# 入力: model(torch.nn.Module), x(torch.Tensor), save_root(str), pick_fn(callable|None), max_ch(int), cmap(str|None)
# 出力: 画像ファイル群(None)
def save_all_fmap(model, x, save_root, pick_fn=None, max_ch=64, cmap=None):
    import math
    from pathlib import Path
    if pick_fn is None:
        pick_fn = lambda m: isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d))
    if cmap is None:
        cmap = FEATUREMAP_CMAP
    model.eval()
    fmap_buf, layer_names, hooks = [], [], []
    def _register(mod, name):
        def _hook(_, __, out):
            fmap_buf.append(out.detach().cpu())
            layer_names.append(name)
        hooks.append(mod.register_forward_hook(_hook))
    for n, m in model.named_modules():
        if pick_fn(m):
            _register(m, n)
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)
    for fmap, lname in zip(fmap_buf, layer_names):
        fmap = fmap[0]  # (C,H,W)
        # レイヤ種別をファイル名に付与
        if 'conv' in lname.lower():
            lname_out = lname.replace('.', '_') + '_Conv'
        elif 'relu' in lname.lower():
            lname_out = lname.replace('.', '_') + '_ReLU'
        elif 'batchnorm' in lname.lower():
            lname_out = lname.replace('.', '_') + '_BatchNorm'
        elif 'maxpool' in lname.lower():
            lname_out = lname.replace('.', '_') + '_MaxPool'
        else:
            lname_out = lname.replace('.', '_')
        n_show = min(max_ch, fmap.shape[0])
        cols = math.ceil(math.sqrt(n_show))
        rows = math.ceil(n_show / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axs = np.ravel(axs)
        for i in range(n_show):
            axs[i].imshow(fmap[i], cmap=cmap)
            axs[i].axis("off")
        for i in range(n_show, rows*cols):
            axs[i].axis("off")
        plt.tight_layout(pad=2.0)  # 隙間を広げる
        fig.savefig(root / f"{lname_out}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ----------------------------
# Predict and plot result
# ----------------------------
#? 画像1枚の推論＋予測点を描画した画像を保存する
# 入力: model(torch.nn.Module), img_path(str), device(str|torch.device)
# 出力: (pts_orig(list[tuple[float,float]]), out_path(str))
def predict_and_plot(model, img_path, device):
    import os
    import numpy as np
    from PIL import Image, ImageDraw
    import torchvision.transforms as T

    model.eval()

    # ── 画像の前処理 ──
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    orig = Image.open(img_path).convert('RGB')
    w0, h0 = orig.size  # 元画像の幅・高さ
    x = transform(orig).unsqueeze(0).to(device)

    # ── 推論 ──
    with torch.no_grad():
        out = model(x)
        # out[0, :8] は「0～1 の正規化座標」が返ってくる前提
        pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)

    # ── normalized → pixel に戻す ──
    pts_orig = []
    for (x_n, y_n) in pts_norm:
        x_px = x_n * w0
        y_px = y_n * h0
        pts_orig.append((x_px, y_px))

    # ── 出力画像フォルダを作る ──
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    # ── 予測点を描画 ──
    draw_img = orig.copy()
    draw = ImageDraw.Draw(draw_img)
    colors = ['red', 'blue', 'green', 'yellow']
    for (x_, y_), c in zip(pts_orig, colors):
        r = 5
        draw.ellipse((x_-r, y_-r, x_+r, y_+r), fill=c)

    # ── 4点間を結ぶ線を引く ──
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts_orig
    draw.line([(x1, y1), (x2, y2)], fill="white", width=2)
    draw.line([(x1, y1), (x3, y3)], fill="white", width=2)
    draw.line([(x1, y1), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x4, y4)], fill="white", width=2)
    draw.line([(x2, y2), (x3, y3)], fill="white", width=2)
    draw.line([(x3, y3), (x4, y4)], fill="white", width=2)

    out_path = os.path.join(IMG_OUT_DIR, os.path.basename(img_path))
    draw_img.save(out_path)

    return pts_orig, out_path


#? DataLoaderワーカーの乱数初期化
# 入力: worker_id(int), rank(int), seed(int)
# 出力: なし(None)
def worker_init_fn(worker_id, rank=0, seed=42):
    np.random.seed(seed + worker_id + rank)
    random.seed(seed + worker_id + rank)

#? DataLoader用collate関数（画像・座標・マスクをバッチ化）
# 入力: batch(list[tuple[Tensor, Tensor, Tensor]])
# 出力: (imgs(torch.Tensor), pts(torch.Tensor), masks(torch.Tensor))
def yolo_dataset_collate(batch):
    # 画像, 座標, マスク, ゲート存在ラベルをバッチ化
    imgs, pts, masks, gate_exists = zip(*batch)
    return torch.stack(imgs), torch.stack(pts), torch.stack(masks), torch.stack(gate_exists)

def heatmap(model, image, save_file_name, target_layer):
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import os

    features = {}
    grads = {}

    def forward_hook(module, input, output):
        features['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        grads['value'] = grad_output[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    y_pred = model(image)
    pred_index = torch.argmax(y_pred)
    model.zero_grad()
    y_pred[0][pred_index].backward()

    # Grad-CAM重み計算（標準的な方法）
    gradients = grads['value']  # (B, C, H, W)
    activations = features['value']  # (B, C, H, W)
    alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    grad_cam = (alpha * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    grad_cam = F.relu(grad_cam)
    grad_cam = grad_cam[0, 0].cpu().numpy()
    grad_cam -= grad_cam.min()
    grad_cam /= (grad_cam.max() + 1e-8)

    # 入力画像サイズにリサイズ
    _, _, H, W = image.shape
    grad_cam_img = cv2.resize(grad_cam.astype(np.float32), (W, H))

    # カラーマップ適用
    cm = plt.get_cmap('jet')
    heatmap = cm(grad_cam_img)[:, :, :3]  # RGBA→RGB
    heatmap = np.uint8(heatmap * 255)

    # 入力画像を0-1正規化＆3ch化
    img1 = image[0].detach().cpu().numpy()  # (C, H, W)
    if img1.shape[0] == 1:
        img1 = np.repeat(img1, 3, axis=0)  # 1ch→3ch
    img1 = np.transpose(img1, (1, 2, 0))  # (H, W, C)
    img1 -= img1.min()
    img1 /= (img1.max() + 1e-8)
    img1 = np.uint8(img1 * 255)

    # 合成
    if heatmap.shape != img1.shape:
        heatmap = cv2.resize(heatmap, (img1.shape[1], img1.shape[0]))
    alpha = 0.5
    grad_cam_image = cv2.addWeighted(img1, alpha, heatmap, 1 - alpha, 0)

    # 保存
    im = Image.fromarray(grad_cam_image)
    os.makedirs(HEATIMG_OUT_DIR, exist_ok=True)
    out_path = os.path.join(HEATIMG_OUT_DIR, f"{save_file_name}_heatmap.png")
    im.save(out_path)
    print(f"ヒートマップ画像を保存しました: {out_path}")

    handle_f.remove()
    handle_b.remove()
#? カメラのライブ映像から推論し、結果を表示する
# 入力: model(torch.nn.Module), device(str|torch.device), out_dir(str)
# 出力: なし
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

def predict_from_camera(model, device, out_dir):
    import time
    model.eval()
    transform = T.Compose([
        T.Resize((160, 160)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor()
    ])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return
    print("'q'キーで終了します")
    point_labels = POINT_LABEL  # ['1', '2', '3', '4']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得失敗")
            break

        # 推論用に 160x160 グレースケール画像を作成
        frame_resized = cv2.resize(frame, (160, 160))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(frame_gray)
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            # 正規化座標として受け取り
            pts_norm = out[0, :8].cpu().numpy().reshape(-1, 2)  # shape=(4,2)
            gate_logit = out[0, 8]
            gate_prob = torch.sigmoid(gate_logit).item()

        # 表示用フレーム（元映像をカラーで640x640に拡大）
        frame_color = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        # 正規化座標を640x640にマッピング
        pts_disp = []
        for (x_n, y_n) in pts_norm:
            x_px = int(x_n * 640)
            y_px = int(y_n * 640)
            pts_disp.append((x_px, y_px))

        # 描画（OpenCV で円をカラーで描画）
        draw_colors = [(0,0,255), (255,0,0), (0,255,0), (0,255,255)]  # 赤, 青, 緑, 黄
        for (x_, y_), c in zip(pts_disp, draw_colors):
            cv2.circle(frame_color, (x_, y_), 12, c, -1)

        # 4点間を結ぶ「外周」線のみ描画（正方形の4辺）
        if len(pts_disp) == 4:
            cv2.line(frame_color, pts_disp[0], pts_disp[1], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[1], pts_disp[3], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[3], pts_disp[2], (255,255,255), 3)
            cv2.line(frame_color, pts_disp[2], pts_disp[0], (255,255,255), 3)

        # Gate 確率を表示（左上に）
        cv2.putText(frame_color, f"GateProb: {gate_prob:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.imshow('Gate Live Predict (Color 640x640)', frame_color)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def heatmap_chat(model, image_tensor, save_path, target_layer):
    """
    Grad-CAMによるヒートマップを生成し、入力画像に重ねて保存する関数。
    model: PyTorchモデル
    image_tensor: (1, C, H, W) のテンソル
    save_path: 保存先パス
    target_layer: ヒートマップを可視化したい畳み込み層（例: model.conv4b）
    """
    model.eval()
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # Hook登録
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    image_tensor = image_tensor.requires_grad_()
    output = model(image_tensor)
    # 出力が複数クラスの場合は最大スコアのクラスを対象
    if output.ndim == 2 or output.ndim == 1:
        class_idx = output.argmax() if output.ndim == 2 else 0
        score = output[0, class_idx] if output.ndim == 2 else output[class_idx]
    else:
        score = output
    model.zero_grad()
    score.backward(retain_graph=True)

    # Grad-CAM重み計算
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    grad_cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    grad_cam = F.relu(grad_cam)
    grad_cam = grad_cam[0, 0].cpu().numpy()
    grad_cam -= grad_cam.min()
    grad_cam /= (grad_cam.max() + 1e-8)

    # 入力画像サイズにリサイズ
    _, _, H, W = image_tensor.shape
    grad_cam_img = Image.fromarray(np.uint8(grad_cam * 255)).resize((W, H), resample=Image.BILINEAR)
    grad_cam_img = np.array(grad_cam_img)

    # カラーマップ適用（jet）
    import matplotlib.cm as cm
    heatmap = cm.jet(grad_cam_img / 255.0)[:, :, :3]  # RGBA→RGB
    heatmap = np.uint8(heatmap * 255)

    # 入力画像をPILで取得
    img = image_tensor[0].detach().cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW→HWC
    elif img.shape[0] == 1:
        img = img[0]  # (1, H, W) → (H, W)
    img -= img.min()
    img /= (img.max() + 1e-8)
    img = np.uint8(img * 255)
    img = np.squeeze(img)  # (H, W, 1)や(1, H, W)→(H, W)
    img_pil = Image.fromarray(img)
    # 画像モードを統一
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    # ヒートマップと重ね合わせ
    heatmap_pil = Image.fromarray(heatmap).convert('RGB')
    # サイズを統一
    if heatmap_pil.size != img_pil.size:
        heatmap_pil = heatmap_pil.resize(img_pil.size, resample=Image.BILINEAR)
    overlay = Image.blend(img_pil, heatmap_pil, alpha=0.5)
    overlay.save(save_path)

    # Hook解除
    fwd_handle.remove()
    bwd_handle.remove()

    return overlay




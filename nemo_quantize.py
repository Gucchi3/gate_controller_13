import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "nemo")))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as T
import nemo
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import json
import logging
import random

from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from nets.net1_ex import net1_ex
from config_ex import (
    PARENT_DIR, IMG_DIR, DATASET_DIR, JSON_DIR, OUT_DIR, IMG_OUT_DIR, INPUT_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EVAL_PERIOD, NUM_WORKERS, DIST_THRESH, PRED_CKPT,
    HEATMAP_CMAP, HEATMAP_IMG_CMAP, FEATUREMAP_CMAP, MEAN_ERROR_CURVE_COLOR, POINT_ERROR_COLORS,SHOW_SUMMAR,
    AUGMENTATION_ENABLED, FLIP_PROB, ROTATE_PROB, ROTATE_DEGREE, SCALE_PROB, SCALE_RANGE,
    SAVE_INPUT_IMG, INPUT_IMG_DIR,
    CONTRAST_PROB, CONTRAST_RANGE, BRIGHTNESS_PROB, BRIGHTNESS_RANGE, SHARPNESS_PROB, SHARPNESS_RANGE,POINT_LABEL
)
from utils_ex import split_dataset, mean_error,max_error, accuracy_at_threshold, plot_heatmap, plot_heatmap_for_image, plot_metric_curve, predict_with_features, save_all_fmap, predict_and_plot,worker_init_fn,yolo_dataset_collate


input_size=160
#-------------------------------------------------------------------------------------------
#　情報
#-------------------------------------------------------------------------------------------
def information():#コメントを折りたたむためだけの意味のない関数
    #! nemo.nemo.graph.py：141~145行目
    #?      渡す引数から「_retain_param_name=True」を削除
    #*      pytorchのversion違いによって実行できなかったが、これを削除したら動いた。影響は知らん。

    #!nemo.nemo.graph.py：49~50行
    #?      「is not」はpythonでは使えないらしい。
    #*      「!=」に変更した。
    return 0

    #!nemo.nemo.qunat.pact.py 56行目
    #?warning排除
    #*はい


#------------------------------
#モデル設定
#------------------------------
model = net1_ex()
#------------------------------
#ログ設定
#------------------------------
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Training / Validation loops
# ----------------------------
#? 1エポック分の学習を実行し、平均lossを返す
# 入力: model(torch.nn.Module), loader(DataLoader), optimizer(torch.optim.Optimizer), device(torch.device), writer(SummaryWriter), epoch(int)
# 出力: 平均loss (float)
def train_one_epoch(model, loader, optimizer, device, writer, epoch):
    global input_size
    printed = False
    model.train()
    running_loss = 0.0
    img_save_count = 0
    from config_ex import GATE_EXIST_LOSS_WEIGHT
    for imgs, targets, masks, gate_exists in tqdm(loader, desc=f"Epoch {epoch} Train", leave=False):
        imgs, targets, masks, gate_exists = imgs.to(device), targets.to(device), masks.to(device), gate_exists.to(device)
        out = model(imgs)  # [B, 9]
        preds = out[:, :8]  # [B,8] 4点座標
        gate_logits = out[:, 8]  # [B] ゲート存在logit
        print(gate_logits)
               # --- バッチの一部をプリントしてみる ---
        if epoch==1 or not printed:
            print("\n=== Debug: Epoch 1, Batch 1 の preds と targets (最初の1サンプルだけ) ===")
            print("preds[0]:", preds[0].detach().cpu().numpy().round(3))
            print("target[0]:", (targets[0] * masks[0]).detach().cpu().numpy().round(3), "(mask をかけた座標)")
            print("mask[0] :", masks[0].detach().cpu().numpy())
            print("gate_exists[0]:", gate_exists[0].item())
            printed=True
            
  
        # 座標損失
        loss_coords = F.smooth_l1_loss(preds * masks, targets * masks, reduction='sum') / (masks.sum() + 1e-6)
        loss_coords = input_size * loss_coords
        # ゲート存在損失
        loss_gate = F.binary_cross_entropy_with_logits(gate_logits.squeeze(), gate_exists)
        loss = 4 * loss_coords + GATE_EXIST_LOSS_WEIGHT * loss_gate
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    avg = running_loss / len(loader.dataset)
    writer.add_scalar('Loss/train', avg, epoch)
    return avg

#? 検証データで評価し、平均lossを返す
# 入力: model(torch.nn.Module), loader(DataLoader), device(torch.device), writer(SummaryWriter), epoch(int), tag(str)
# 出力: 平均loss (float)
def validate(model, loader, device, writer, epoch, tag='val'):
    global input_size
    model.eval()
    running_loss = 0.0
    from config_ex import GATE_EXIST_LOSS_WEIGHT
    with torch.no_grad():
        for imgs, targets, masks, gate_exists in tqdm(loader, desc=f"{tag} {epoch}", leave=False):
            imgs, targets, masks, gate_exists = imgs.to(device), targets.to(device), masks.to(device), gate_exists.to(device)
            out = model(imgs)  # [B, 9]
            preds = out[:, :8]  # [B,8]
            gate_logits = out[:, 8]  # [B]
            loss_coords = F.smooth_l1_loss(preds * masks, targets * masks, reduction='sum') / (masks.sum() + 1e-6)
            loss_coords = input_size * loss_coords
            loss_gate = F.binary_cross_entropy_with_logits(gate_logits.squeeze(), gate_exists)
            loss = 4*loss_coords + GATE_EXIST_LOSS_WEIGHT * loss_gate
            running_loss += loss.item() * imgs.size(0)
    avg = running_loss / len(loader.dataset)
    writer.add_scalar(f'Loss/{tag}', avg, epoch)
    return avg

# ---------------------------
# Dataset with mask
# ----------------------------
#? 4点コーナー推論用データセット（LabelMe形式対応）
# 入力: json_paths (list[str]), img_dir (str), input_size (tuple[int,int]), transforms (callable|None)
# 出力: Dataset (画像,座標,マスク)
class LabelMeCornerDataset(Dataset):
    REQUIRED_LABELS = POINT_LABEL

    def __init__(self, json_paths, img_dir, input_size=INPUT_SIZE, transforms=None, is_train=False):
        self.items = []
        if json_paths:
            for jp in json_paths:
                try:
                    data = json.load(open(jp, 'r'))
                    pts_map = {}
                    for shape in data['shapes']:
                        if shape.get('shape_type') == 'point':
                            lbl = shape.get('label')
                            if lbl in self.REQUIRED_LABELS:
                                pts_map[lbl] = shape['points'][0]
                    self.items.append({'image': data['imagePath'], 'pts': pts_map})
                except Exception as e:
                    logger.error(f"JSON load fail {jp}: {e}")
        else:
            # jsonがない場合: img_dir内の画像を全てitemsに追加（ptsは空dict）
            img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(" # jsonがない場合: img_dir内の画像を全てitemsに追加（ptsは空dict）→ 実行")
            for imgf in img_files:
                self.items.append({'image': imgf, 'pts': {}})
        self.img_dir = img_dir
        self.input_size = input_size
        self.transforms = transforms or T.Compose([
            T.Resize(input_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):  # --- コード的にはここがメイン ---
        rec = self.items[idx]
        img_path = os.path.join(self.img_dir, rec['image'])

        try:
            img = Image.open(img_path)
            # --- sRGB→リニア ガンマ補正 ---
            if img.mode != 'L':
                img_np = np.asarray(img).astype(np.float32) / 255.0
                # sRGB→リニア
                def srgb_to_linear(x):
                    mask = x <= 0.04045
                    return np.where(mask, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    img_np = srgb_to_linear(img_np)
                    # 再度8bitに戻してPIL Image化
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_np, mode='RGB')
                elif img_np.ndim == 2:
                    img_np = srgb_to_linear(img_np)
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_np, mode='L')
            # --- グレースケール変換 ---
            img = img.convert('L')  # --- 'L' は grayscale のこと ---
        except Exception as e:
            logger.error(f"Image load fail {img_path}: {e}")
            img = Image.new('L', self.input_size, (0))

        w0, h0 = img.size  # width, height  (通常 input_size=(160,160) のはず)

        # --- jsonがない or ptsが空dictの場合 ---
        if not rec['pts']:
            pts = np.zeros((len(self.REQUIRED_LABELS), 2), dtype=np.float32)
            mask = np.zeros(len(self.REQUIRED_LABELS) * 2, dtype=np.float32)
            gate_exist = 0.0
        else:
            pts = []
            mask = []
            for lbl in self.REQUIRED_LABELS:
                if lbl in rec['pts']:
                    x, y = rec['pts'][lbl]
                    pts += [x, y]
                    mask += [1.0, 1.0]
                else:
                    pts += [0.0, 0.0]
                    mask += [0.0, 0.0]
            pts = np.array(pts, dtype=np.float32).reshape(len(self.REQUIRED_LABELS), 2)
            #!---------------------------------------------------------
            #! 　　　　　0, 1 列目
            #!  0 行目 [x, y],　
            #!  1 行目 [x, y],　
            #!  2 行目 [x, y],
            #!  3 行目 [x, y],
            #!---------------------------------------------------------
            #!-----------------------------------------------------------------
            #! | 書き方            | 返り値の意味
            #! | -------------- | --------------------------------------------
            #! | `pts[1, 0]`    | 行 1・列 0 → right の x 座標（スカラー）
            #! | `pts[1, 1]`    | 行 1・列 1 → right の y 座標（スカラー）
            #! | `pts[1,2]`     | 行 1・列 2 → 単一要素。ここでは列 2 がないのでエラー
            #! | `pts[[1,2]]`   | 行 1 と行 2 → `[[xr,yr],[xl,yl]]`（2×2 の配列)
            #! | `pts[[1,2],0]` | 行 1,2 の x 列 → `[xr, xl]`（形状 (2,) の配列）
            #! | `pts[[1,2],1]` | 行 1,2 の y 列 → `[yr, yl]`
            #!-----------------------------------------------------------------

            #? augmentations (trainのみ)
            # !--- 画像拡張系の関数の追加時は要検証!! ---
            if self.is_train and AUGMENTATION_ENABLED:
                #? 1. 左右反転
                if random.random() < FLIP_PROB:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)  # --- 画像の反転 ---
                    pts[:, 0] = w0 - 1 - pts[:, 0]  # --- w0 には 160 が格納されている。画像は 0~159 のため、w0 -1 -pts で反転 ---
                    # ラベル順序も反転（top, right, left, bottom → top, left, right, bottom）
                    pts[[1, 2]] = pts[[2, 1]]

                #? 2. 拡大縮小
                if random.random() < SCALE_PROB:
                    scale = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])   # 拡大縮小率をランダムに選択
                    nw, nh = int(w0 * scale), int(h0 * scale)                # 新しい画像サイズを計算
                    img = img.resize((nw, nh), resample=Image.BILINEAR)      # 画像をリサイズ（バイリニア補完）
                    pts = pts * scale                                        # アノテーションも同じ倍率で拡大縮小

                    # 元サイズに戻す（中央切り出し or パディング）
                    if scale >= 1.0:
                        # ── 拡大 → 中央を切り出し ─────────────────────────────
                        left = (nw - w0) // 2
                        upper = (nh - h0) // 2
                        img = img.crop((left, upper, left + w0, upper + h0))
                        pts = pts - [left, upper]

                        # 画像外に出たキーポイントはマスクを 0 に
                        oob = (pts[:, 0] < 0) | (pts[:, 0] >= w0) | (pts[:, 1] < 0) | (pts[:, 1] >= h0)
                        mask_np = np.array(mask, dtype=np.float32).reshape(-1, 2)
                        mask_np[oob, :] = 0.0
                        mask = mask_np.flatten().tolist()
                    else:
                        # ── 縮小 → ランダム配置（四隅 or 中央） ─────────────────
                        new_img = Image.new('L', (w0, h0), 0)  # 160×160 の黒背景を用意

                        # 5 通りからランダムに配置位置 (left, upper) を選択
                        positions = [
                            (0, 0),                         # 左上
                            (w0 - nw, 0),                   # 右上
                            (0, h0 - nh),                   # 左下
                            (w0 - nw, h0 - nh),             # 右下
                            ((w0 - nw) // 2, (h0 - nh) // 2)  # 中央
                        ]
                        left, upper = random.choice(positions)

                        new_img.paste(img, (left, upper))  # リサイズ済み画像を貼り付け
                        img = new_img
                        pts = pts + [left, upper]  # アノテーションをシフト

                        # 縮小配置の場合、画像外に出ることは無いのでマスク更新不要

                #? 3. ランダム回転
                if random.random() < ROTATE_PROB:
                    angle = random.uniform(-ROTATE_DEGREE, ROTATE_DEGREE)  # 回転角をランダムに決定
                    img = img.rotate(angle, resample=Image.BILINEAR)  # 画像回転、バイリニア補完
                    # 座標も回転
                    cx, cy = w0 / 2, h0 / 2
                    theta = np.deg2rad(angle)
                    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
                    pts = pts - [cx, cy]
                    pts = np.dot(pts, rot_mat.T)  # 行列計算
                    pts = pts + [cx, cy]
                    oob = (pts[:, 0] < 0) | (pts[:, 0] >= w0) | (pts[:, 1] < 0) | (pts[:, 1] >= h0)
                    mask_np = np.array(mask, dtype=np.float32).reshape(-1, 2)
                    mask_np[oob, :] = 0.0
                    mask = mask_np.flatten().tolist()

                #? 4. コントラスト変換
                if random.random() < CONTRAST_PROB:
                    from PIL import ImageEnhance
                    factor = random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1])
                    img = ImageEnhance.Contrast(img).enhance(factor)  # コントラスト変換

                #? 5. 明るさ変換
                if random.random() < BRIGHTNESS_PROB:
                    from PIL import ImageEnhance
                    factor = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
                    img = ImageEnhance.Brightness(img).enhance(factor)  # 明るさ変換

                #? 6. シャープネス変換
                if random.random() < SHARPNESS_PROB:
                    from PIL import ImageEnhance
                    factor = random.uniform(SHARPNESS_RANGE[0], SHARPNESS_RANGE[1])
                    img = ImageEnhance.Sharpness(img).enhance(factor)  # シャープネス変換

        # 変換後のリサイズ・グレースケール・Tensor化
        tensor_img = self.transforms(img)


        # 「正規化ラベル」を作成する部分
        pts_normalized = []
        for x, y in pts:
            x_norm = x / float(w0)
            y_norm = y / float(h0)
            pts_normalized += [x_norm, y_norm]

        # ゲート存在ラベル（1点でもアノテーションがあれば 1.0、なければ 0.0）
        gate_exist = 1.0 if np.any(np.array(mask) > 0.0) else 0.0

        # ファイル名と正解座標を表示
        #print(f"{rec['image']}: top{tuple(pts[0])}, right{tuple(pts[1])}, left{tuple(pts[2])}, bottom{tuple(pts[3])}")
        return (
            tensor_img,
            torch.tensor(pts_normalized, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(gate_exist, dtype=torch.float32)
        )


        #? 【学習・検証機能（mode==1）】
        # main
        #  └─ train_dataset = LabelMeCornerDataset(...)   # ← ここでインスタンス化
        #  └─ val_dataset   = LabelMeCornerDataset(...)
        #  └─ test_dataset  = LabelMeCornerDataset(...)
        #      ├─ DataLoaderでラップ
        #      │   └─ for imgs, targets, masks in DataLoader(...):
        #      │         └─ __getitem__が呼ばれる（画像・座標・マスクを返す）
        #      ├─ train_one_epoch / validate / mean_error などでデータが使われる

        #? 【推論機能（mode==2）】
        # main
        #  └─ 入力画像パス取得（ユーザー入力）
        #  └─ model.load_state_dict
        #  └─ predict_with_features(model, img_path, device, out_dir)
        #      └─ 画像1枚を直接PILで読み込み、前処理・推論
        #      └─ ※通常はLabelMeCornerDatasetは使われない

#----------------------------
#　utils
#----------------------------
def remove_txt():
    # Removing old activations and inputs

    directory = 'log'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if (file.startswith("out_layer") or file.startswith("input.txt"))]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)

#----------------------------
#　test
#----------------------------

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
            # print("================================================================================================")
            # print(f"targets：{targets}\n\npreds：{out}")
            # print("================================================================================================")

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


#?-------------------------------------------------------------------------------------------------------
#?　--メインステップ--
#?-------------------------------------------------------------------------------------------------------
def main():
    global model
    #---------------------------
    # デバイス設定、その他
    #---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    start_time = datetime.now().strftime('%Y%m%d-%H%M')#log保存用のフォルダ作成
    session_dir = os.path.join(OUT_DIR, start_time)
    os.makedirs(session_dir, exist_ok=True)
    val_losses = []
    train_losses = []
    train_js, val_js, test_js = split_dataset(JSON_DIR)
    train_dataset = LabelMeCornerDataset(train_js, DATASET_DIR, is_train=True)
    val_dataset   = LabelMeCornerDataset(val_js,   DATASET_DIR, is_train=False)
    test_dataset  = LabelMeCornerDataset(test_js,  DATASET_DIR, is_train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
        persistent_workers=True,
        prefetch_factor=6,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=42)
        )
    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    collate_fn=yolo_dataset_collate,
    persistent_workers=True,
    prefetch_factor=6,
    worker_init_fn=partial(worker_init_fn, rank=0, seed=42)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
        persistent_workers=True,
        prefetch_factor=6,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=42)
    )
    loss_curve_path = os.path.join(session_dir, 'loss_curve.png')
    writer    = SummaryWriter(log_dir=session_dir)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    #---------------------------
    #　モデルに学習済み重みを適用
    #---------------------------
    ckpt = torch.load("log/best.pth", map_location="cpu")
    # DataParallelで保存されていた場合はキーの先頭に'module.'が付いていることがあるので注意
    if all(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    #------------------------------
    #　テストを実行 #?（getした重みやモデルがちゃんと動作するかの確認）
    #------------------------------
    remove_txt()
    mean_err, errors, point_errors = mean_error(model, test_loader, device)
    max_err = max_error(errors)
    acc5 = accuracy_at_threshold(errors, 5.0)
    acc10 = accuracy_at_threshold(errors, 10.0)
    print("-----------------------------------------------------------------------------------------------")
    print(f"[Test][original] mean_err: {mean_err:.4f}, max_err: {max_err:.4f}, acc@5px: {acc5:.4f}, acc@10px: {acc10:.4f}")
    print("-----------------------------------------------------------------------------------------------")

#!-----------------------------------------------------------------------------------------------------------------------------------------------
    #?-----------------------------
    #?　nemoを用いてキャリブレーション
    #?-----------------------------
    model = nemo.quantize_pact(model, dummy_input=torch.randn(1,1,160,160).to(device))#note:簡単だね(⋈◍＞◡＜◍)。✧♡

    # precision = {
    #     "conv1":   {"W_bits": 15},
    #     "relu1":   {"x_bits": 16},
    #     "conv1b":  {"W_bits": 15},
    #     "relu1b":  {"x_bits": 16},
    #     "conv2":   {"W_bits": 15},
    #     "relu2":   {"x_bits": 16},
    #     "conv2b":  {"W_bits": 15},
    #     "relu2b":  {"x_bits": 16},
    #     "conv3":   {"W_bits": 15},
    #     "relu3":   {"x_bits": 16},
    #     "conv3b":  {"W_bits": 15},
    #     "relu3b":  {"x_bits": 16},
    #     "conv4":   {"W_bits": 15},
    #     "relu4":   {"x_bits": 16},
    #     "conv4b":  {"W_bits": 15},
    #     "relu4b":  {"x_bits": 16},
    #     "fc":      {"W_bits": 15}
    # }



    #model.change_precision(bits=1,min_prec_dict=precision, scale_weights=True, scale_activations=True)

    # #?-----------------------------
    # #?　統計量取得
    # #?-----------------------------
    # with model.statistics_act():
    #     _ = mean_error(model, train_loader, device)
    # model.reset_alpha_act()
    
    # #?------------------------------
    # #?　モデルからbiasを削除
    # #?------------------------------
    # model.remove_bias()
    
    # #?-----------------------------
    # #?　疑似量子化したものをテスト
    # #?-----------------------------
    # # remove_txt()
    # mean_err, errors, point_errors = mean_error(model, test_loader, device)
    # max_err = max_error(errors)
    # acc5 = accuracy_at_threshold(errors, 5.0)
    # acc10 = accuracy_at_threshold(errors, 10.0)
    # print("-----------------------------------------------------------------------------------------------")
    # print(f"[Test][15-16quantize] mean_err: {mean_err:.4f}, max_err: {max_err:.4f}, acc@5px: {acc5:.4f}, acc@10px: {acc10:.4f}")
    # print("-----------------------------------------------------------------------------------------------")


    #?------------------------------
    #?　さらにキャリブレーション
    #?------------------------------
    precision = {
        "conv1":   {"W_bits": 7},
        "relu1":   {"x_bits": 8},
        "conv1b":  {"W_bits": 7},
        "relu1b":  {"x_bits": 8},
        "conv2":   {"W_bits": 7},
        "relu2":   {"x_bits": 8},
        "conv2b":  {"W_bits": 7},
        "relu2b":  {"x_bits": 8},
        "conv3":   {"W_bits": 7},
        "relu3":   {"x_bits": 8},
        "conv3b":  {"W_bits": 7},
        "relu3b":  {"x_bits": 8},
        "conv4":   {"W_bits": 7},
        "relu4":   {"x_bits": 8},
        "conv4b":  {"W_bits": 7},
        "relu4b":  {"x_bits": 8},
        "fc":      {"W_bits": 7}
    }

    model.change_precision(bits=1, min_prec_dict=precision, scale_weights=True, scale_activations=True)

    #?-----------------------------
    #?　重みのclip paramの調整...??
    #?-----------------------------
    with model.statistics_act():
        _ = mean_error(model, test_loader, device)
    model.reset_alpha_act()
    
     #?------------------------------
     #?　モデルからbiasを削除
     #?------------------------------
    model.remove_bias()
    
    #-----------------------------
    #　test
    #-----------------------------
   # remove_txt()
    mean_err, errors, point_errors = mean_error(model, test_loader, device)
    max_err = max_error(errors)
    acc5 = accuracy_at_threshold(errors, 5.0)
    acc10 = accuracy_at_threshold(errors, 10.0)
    print("-----------------------------------------------------------------------------------------------")
    print(f"[Test][7-8quantize] mean_err: {mean_err:.4f}, max_err: {max_err:.4f}, acc@5px: {acc5:.4f}, acc@10px: {acc10:.4f}")
    print("-----------------------------------------------------------------------------------------------")
    #model.load_state_dict(torch.load("./best.pth",map_location=torch.device("cuda")))
    


    # #?-----------------------------
    # #?　BN層の折り畳み
    # #?-----------------------------
    # model.fold_bn()
    # model.reset_alpha_weights()
    model = nemo.transform.bn_to_identity(model)
    #?-------------------------------
    #?　qd_stage
    #?-------------------------------
    # BN層をIdentityに置換
    # print("==== model modules before qd_stage ====")
    # for name, module in model.named_modules():
    #     print(name, type(module))
    # print("=======================================")
    model.qd_stage(eps_in=1./255)


    #-----------------------------
    #　test
    #-----------------------------
    #remove_txt()
    mean_err, errors, point_errors = mean_error(model, test_loader, device)
    max_err = max_error(errors)
    acc5 = accuracy_at_threshold(errors, 5.0)
    acc10 = accuracy_at_threshold(errors, 10.0)
    print("-----------------------------------------------------------------------------------------------")
    print(f"[Test][qd_stage] mean_err: {mean_err:.4f}, max_err: {max_err:.4f}, acc@5px: {acc5:.4f}, acc@10px: {acc10:.4f}")
    print("-----------------------------------------------------------------------------------------------")


    #------------------------------
    #　id_stage
    #------------------------------
    model.id_stage()
   # print(model)


    #?-----------------------------
    #?　id_stage用の関数を定義
    #?-----------------------------
    #remove_txt()
    from config_ex import POINT_LABEL

    def mean_error_int8(model, loader, device, integer=False):
        global input_size
        model.eval()
        errors = []
        point_labels = POINT_LABEL
        point_errors = {lbl: [] for lbl in point_labels}
        with torch.no_grad():
            for imgs, targets, masks, gate_exists in loader:
                if integer:
                    imgs = (imgs * 255).round().clamp(0, 255)
                    imgs = imgs.type(torch.uint8).type(torch.float).to(device)
                    imgs = imgs / 255.0  # int8入力時も0-1スケールに戻す
                    targets = targets.to(device)
                    masks = masks.to(device)
                else:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    masks = masks.to(device)
                out = model(imgs)  # [B,9]
                print(f"targes：{targets}\n\n　preds：{out}")
                preds = out[:, :8].cpu().numpy()  # [B,8]
                tars  = targets.cpu().numpy()
                ms    = masks.cpu().numpy()
                for p, t, m in zip(preds, tars, ms):
                    for i, lbl in zip(range(0, len(point_labels)*2, 2), point_labels):
                        if m[i] == 0: continue
                        gt = np.array([t[i], t[i+1]])
                        pr = np.array([p[i], p[i+1]])
                        err = np.linalg.norm(gt - pr)
                        err *= input_size
                        errors.append(err)
                        point_errors[lbl].append(err)
        return (np.mean(errors) if errors else 0.0, errors, point_errors)


    #?-----------------------------
    #?　test (id_stage, int8入力)
    #?-----------------------------
   # remove_txt()
    mean_err, errors, point_errors = mean_error_int8(model, test_loader, device, integer=True)
    max_err = max_error(errors)
    acc5 = accuracy_at_threshold(errors, 5.0)
    acc10 = accuracy_at_threshold(errors, 10.0)
    print("-----------------------------------------------------------------------------------------------")
    print(f"[Test][id_stage][int8input] mean_err: {mean_err:.4f}, max_err: {max_err:.4f}, acc@5px: {acc5:.4f}, acc@10px: {acc10:.4f}")
    print("-----------------------------------------------------------------------------------------------")


    #?----------------------------
    #?　onnx保存
    #?----------------------------
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    nemo.utils.export_onnx("network_nemo.onnx", model, model, (1, 160, 160) ,verbose=False)


if __name__ == '__main__':
    main()


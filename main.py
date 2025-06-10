
#!---------------------------------
#!---折り畳み機能を使って折りたたむこと
#!---------------------------------

#?---------------------------------------------------------------------------------------
#?　import・セッティング
#?---------------------------------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import json
import logging
import math
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
from torchinfo import summary

from config import (
    PARENT_DIR, IMG_DIR, DATASET_DIR, JSON_DIR, OUT_DIR, IMG_OUT_DIR, INPUT_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EVAL_PERIOD, NUM_WORKERS, DIST_THRESH, PRED_CKPT,
    HEATMAP_CMAP, HEATMAP_IMG_CMAP, FEATUREMAP_CMAP, MEAN_ERROR_CURVE_COLOR, POINT_ERROR_COLORS,SHOW_SUMMAR,
    AUGMENTATION_ENABLED, FLIP_PROB, ROTATE_PROB, ROTATE_DEGREE, SCALE_PROB, SCALE_RANGE,
    SAVE_INPUT_IMG, INPUT_IMG_DIR,
    CONTRAST_PROB, CONTRAST_RANGE, BRIGHTNESS_PROB, BRIGHTNESS_RANGE, SHARPNESS_PROB, SHARPNESS_RANGE,POINT_LABEL
)
from utils import split_dataset, mean_error, max_error, accuracy_at_threshold, plot_heatmap, plot_heatmap_for_image,  predict_with_features, save_all_fmap, predict_and_plot,worker_init_fn,yolo_dataset_collate
from nets.net1 import net1_ex

input_size = INPUT_SIZE[0]

#!------------------------------
NETWORK = net1_ex()
#!------------------------------

REQUIRED_LABELS = POINT_LABEL


#?--------------------------------------------------------------------------------------
#?　Dataset
#? -------------------------------------------------------------------------------------
#? 4点コーナー推論用データセット（LabelMe形式対応）
# 入力: json_paths (list[str]), img_dir (str), input_size (tuple[int,int]), transforms (callable|None)
# 出力: Dataset (画像,座標,マスク)
class LabelMeCornerDataset(Dataset):
    REQUIRED_LABELS = REQUIRED_LABELS

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
                    pts[[1, 3]] = pts[[3, 1]]
                    #* 20250609 

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
                    #!==Pillowは画像の左上を中心として、Θ>0の時に「「「時計回り」」」に回転する。
                            #!===しかし、Pillowではなく数学的に回転させると、画像の左下を中心として、「「「反時計回り」」」に回転する。
                                #!===よって、Pillowと数学的回転では回転方向が違うため、angleに×(-1)をかけて逆方向の角度を指定しなければならない。
                    # 座標も回転
                    cx, cy = w0 / 2, h0 / 2
                    angle_r = math.radians(-angle)#!×(-1)に注意!!!
                    #print(f"angle：{angle}　　angle_rad：{angle_r}") #DEBUG
                    x0 = pts[:, 0] - cx
                    y0 = pts[:, 1] - cy
                    pts[:, 0] =  cx +(x0)*math.cos(angle_r) - (y0)*math.sin(angle_r)
                    pts[:, 1] = cy + (x0)*math.sin(angle_r) + (y0)*math.cos(angle_r)

                    # for i ,(x, y) in enumerate(pts): #maskは__getitemの最後で定義してた(*´ω｀*)
                    #     if x < 0 or input_size <= x:
                    #         mask[i, 0] = 0
                    #     if y < 0 or input_size <= y:(*´ω｀*)
                    #         mask[i, 1] = 0
                        

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
        # --- 入力画像保存（拡張後のTensor→画像） ---
        if SAVE_INPUT_IMG and self.is_train:
            os.makedirs(INPUT_IMG_DIR, exist_ok=True)
            img_np = tensor_img.squeeze().cpu().numpy()  # shape=(H,W)
            img_uint8 = (img_np * 255).clip(0,255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8).convert('RGB')
            draw = ImageDraw.Draw(img_pil)
            pts_arr = np.array(pts).reshape(4,2)
            mask_arr = np.array(mask).reshape(4,2)
            colors = ['red','blue','green','yellow']
            #print(f"[DEBUG] image={rec['image']}, w0={w0}, h0={h0}, img_pil.size={img_pil.size}")
           # print(f"[DEBUG] pts={pts_arr}")
            for j, (pt, m) in enumerate(zip(pts_arr, mask_arr)):
                if m[0] > 0 or m[1] > 0:
                    x = int(pt[0] * img_pil.width / w0)
                    y = int(pt[1] * img_pil.height / h0)
                    r = 4
                    #print(f"[DEBUG] {rec['image']} pt{j}: 元座標=({pt[0]:.2f},{pt[1]:.2f}) → plot=({x},{y}) on image size {img_pil.size}")
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=colors[j])
            img_pil.save(os.path.join(INPUT_IMG_DIR, f"{os.path.splitext(os.path.basename(rec['image']))[0]}_ann.png"))
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


#? -------------------------------------------------------------------------------------
#?　Training / Validation
#? -------------------------------------------------------------------------------------
#? 1エポック分の学習を実行し、平均lossを返す
# 入力: model(torch.nn.Module), loader(DataLoader), optimizer(torch.optim.Optimizer), device(torch.device), writer(SummaryWriter), epoch(int)
# 出力: 平均loss (float)
def train_one_epoch(model, loader, optimizer, device, writer, epoch):
    global input_size
    printed = False
    model.train()
    running_loss = 0.0
    img_save_count = 0
    from config import GATE_EXIST_LOSS_WEIGHT
    for imgs, targets, masks, gate_exists in tqdm(loader, desc=f"Epoch {epoch} Train", leave=False):
        imgs, targets, masks, gate_exists = imgs.to(device), targets.to(device), masks.to(device), gate_exists.to(device)
        out = model(imgs)  # [B, 9]
        preds = out[:, :8]  # [B,8] 4点座標
        gate_logits = out[:, 8]  # [B] ゲート存在logit
        #print(gate_logits)
                # --- バッチの一部をプリントしてみる ---
        # if epoch==1 or not printed:
        #     print("\n=== Debug: Epoch 1, Batch 1 の preds と targets (最初の1サンプルだけ) ===")
        #     print("preds[0]:", preds[0].detach().cpu().numpy().round(3))
        #     print("target[0]:", (targets[0] * masks[0]).detach().cpu().numpy().round(3), "(mask をかけた座標)")
        #     print("mask[0] :", masks[0].detach().cpu().numpy())
        #     print("gate_exists[0]:", gate_exists[0].item())
        #     printed=True
            
  
        # 座標損失
        loss_coords = F.smooth_l1_loss(preds * masks, targets * masks, reduction='sum') / (masks.sum() + 1e-6)
        loss_coords = input_size * loss_coords#160*loss_coords
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
    from config import GATE_EXIST_LOSS_WEIGHT
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


#? --------------------------------------------------------------------------------------
#?　Main
#? --------------------------------------------------------------------------------------
#? メイン処理（学習・推論・可視化などモード選択）
# 入力: なし（コマンドライン入力）
# 出力: なし
def main():
    print("""
==== 機能選択 ====
1. train         : 学習
2. predict       : 画像1枚推論+ヒートマップ+ONNX+特徴マップ
3. batch_predict : imgフォルダ内一括推論
4. camera_live   : カメラライブ推論
5. export_onnx   : モデルONNXエクスポートのみ
==================
""")
    mode = input("番号を入力してください (1-5): ").strip()
    # --- 学習開始時刻のサブフォルダ作成 ---
    if mode == '1':
        start_time = datetime.now().strftime('%Y%m%d-%H%M')#log保存用のフォルダ作成
        session_dir = os.path.join(OUT_DIR, start_time)
        os.makedirs(session_dir, exist_ok=True)
    else:
        session_dir = OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = NETWORK.to(device) 

    if SHOW_SUMMAR == 1:
        if __name__ == '__main__':
            summary(model, input_size=(1, 1, 160, 160), device=device)


    if mode == '1':
        metric_epochs   = []   
        mean_err_values = []   
        point_err_history = {lbl: [] for lbl in REQUIRED_LABELS}  # 各点のエポックごとの誤差推移
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
        
        def exp():#!====DataLoaderの解説====!#
            # ==============================
            # DataLoaderのバッチ化データ構造解説
            # ==============================
            # train_loader, val_loader, test_loader には
            #   (imgs, targets, masks, gate_exists)
            # の4つのTensorが1バッチ分ずつタプルで渡される。
            #
            # - imgs:        torch.Tensor, shape [B, 1, 160, 160]
            #                （Bはバッチサイズ。1chグレースケール画像。0-1正規化済み）
            # - targets:     torch.Tensor, shape [B, 8]
            #                （各画像の4点コーナー座標 [x1,y1,x2,y2,x3,y3,x4,y4]）
            # - masks:       torch.Tensor, shape [B, 8]
            #                （各座標が有効なら1.0, 無効なら0.0。座標ごとに対応）
            # - gate_exists: torch.Tensor, shape [B]
            #                （各画像にゲートが存在すれば1.0, なければ0.0）
            #
            # 例:
            #   for imgs, targets, masks, gate_exists in train_loader:
            #       # imgs.shape        → [B, 1, 160, 160]
            #       # targets.shape     → [B, 8]
            #       # masks.shape       → [B, 8]
            #       # gate_exists.shape → [B]
            #
            # バッチ化はcollate_fn（yolo_dataset_collate）で行われており、
            # 各サンプル（画像, 座標, マスク, ゲート有無）をTensorでまとめている。
            # ==============================
            # DataLoaderの主な引数の解説
            # ==============================
            # - shuffle:
            #     Trueの場合、各エポックごとにデータの順番をランダムにシャッフルする。
            #     学習時（train_loader）で有効にすることで汎化性能が向上しやすい。
            #     検証・テスト時はFalse（順番固定）が一般的。
            #
            # - num_workers:
            #     データ読み込みに使うサブプロセス数。0ならメインプロセスのみ。
            #     1以上にするとデータのロードが並列化され、I/O待ちが減り高速化する。
            #
            # - pin_memory:
            #     Trueの場合、データをCUDA転送しやすいメモリ（ページロックメモリ）に格納。
            #     GPU学習時にデータ転送が高速化される（CPU→GPU）。
            #
            # - drop_last:
            #     データ数がバッチサイズで割り切れない場合、最後の端数バッチを捨てるかどうか。
            #     Trueなら端数を捨て、Falseなら小さいバッチも返す。
            #
            # - persistent_workers:
            #     Trueの場合、エポック間でworkerプロセスを維持し、DataLoaderの再初期化コストを削減。
            #
            # - prefetch_factor:
            #     各workerが先読みするバッチ数。大きくするとI/O待ちが減りやすい。
            #
            # - worker_init_fn:
            #     各workerの初期化時に呼ばれる関数。乱数シードの固定などに使う。
            #
            # ==============================
            return 0
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#--optimizerはAdamを使用してる--#
        writer    = SummaryWriter(log_dir=session_dir)
        best_val  = float('inf')

        train_losses = []
        val_losses = []
        loss_curve_path = os.path.join(session_dir, 'loss_curve.png')

        for ep in range(1, EPOCHS+1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, writer, ep)
            val_loss = validate(model, val_loader, device, writer, ep, 'val')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # --- グラフ描画・保存 ---
            plt.figure()
            plt.plot(range(1, ep+1), train_losses, label='Train Loss')
            plt.plot(range(1, ep+1), val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train/Val Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(loss_curve_path)
            plt.close()
            # --- シンプルなloss表示 ---
            print(f"Epoch {ep}\ntrain loss:{train_loss:.4f} val loss:{val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                model.save_path = PRED_CKPT
                torch.save(model.state_dict(), PRED_CKPT)
                logger.info(f"Saved best model at epoch {ep}")
            if ep % DIST_THRESH == 0:
                # --- 指標計算 ---
                mean_err, errors, point_errors = mean_error(model, test_loader, device)
                max_err = max_error(errors)
                acc5 = accuracy_at_threshold(errors, 5.0)
                acc10 = accuracy_at_threshold(errors, 10.0)
                metric_epochs.append(ep)
                mean_err_values.append(mean_err)
                # 各点の平均誤差も記録
                for i, lbl in enumerate(REQUIRED_LABELS):
                    if len(point_errors[lbl]) > 0:
                        point_err_history[lbl].append(np.mean(point_errors[lbl]))
                    else:
                        point_err_history[lbl].append(0.0)
                # --- mean error + 各点error推移を1つのグラフにまとめて描画 ---
                plt.figure()
                plt.plot(metric_epochs, mean_err_values, color=MEAN_ERROR_CURVE_COLOR, marker='o', label='Mean Error')
                for i, lbl in enumerate(REQUIRED_LABELS):
                    plt.plot(metric_epochs, point_err_history[lbl], color=POINT_ERROR_COLORS[i], marker='o', label=lbl)
                plt.xlabel('Epoch')
                plt.ylabel('Error (px)')
                plt.title('Mean/Pointwise Error over Epochs')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(session_dir, 'mean_point_error_curve.png'))
                plt.close()
                # --- 点ごとの平均誤差グラフ ---
                point_labels = REQUIRED_LABELS
                point_means = [np.mean(point_errors[lbl]) if len(point_errors[lbl]) > 0 else 0.0 for lbl in point_labels]
                plt.figure()
                plt.bar(point_labels, point_means)
                plt.ylabel('Mean Error (px)')
                plt.title('Mean Error by Point')
                plt.tight_layout()
                plt.savefig(os.path.join(session_dir, 'pointwise_mean_error.png'))
                plt.close()
                # --- ヒートマップ出力 ---
                plot_heatmap(model, test_loader, device, session_dir)
                # --- 指標テキスト保存 ---
                with open(os.path.join(session_dir, 'metrics.txt'), 'w') as f:
                    f.write(f"Mean Error: {mean_err:.4f}\n")
                    f.write(f"Max Error: {max_err:.4f}\n")
                    f.write(f"Accuracy@5px: {acc5:.4f}\n")
                    f.write(f"Accuracy@10px: {acc10:.4f}\n")
                    f.write("\nMean Error by Point:\n")
                    for lbl, val in zip(point_labels, point_means):
                        f.write(f"{lbl}: {val:.4f}\n")
                # --- 指標棒グラフ ---
                plt.figure()
                plt.bar(['Mean', 'Max', 'Acc@5px', 'Acc@10px'], [mean_err, max_err, acc5, acc10])
                plt.ylabel('Value')
                plt.title('Test Metrics')
                plt.tight_layout()
                plt.savefig(os.path.join(session_dir, 'metrics_bar.png'))
                plt.close()
                print("=======================================================================")
                print(f"Mean：{mean_err:.4f}　Max：{max_err:.4f}　acc5：{acc5:.4f}　acc10：{acc10:.4f}")
                print("=======================================================================")
                
        # --- 指標計算 ---
        mean_err, errors, point_errors = mean_error(model, test_loader, device)
        max_err = max_error(errors)
        acc5 = accuracy_at_threshold(errors, 5.0)
        acc10 = accuracy_at_threshold(errors, 10.0)
        # --- 点ごとの平均誤差グラフ ---
        point_labels = REQUIRED_LABELS
        point_means = [np.mean(point_errors[lbl]) if len(point_errors[lbl]) > 0 else 0.0 for lbl in point_labels]
        plt.figure()
        plt.bar(point_labels, point_means)
        plt.ylabel('Mean Error (px)')
        plt.title('Mean Error by Point')
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, 'pointwise_mean_error.png'))
        plt.close()
        # --- ヒートマップ出力 ---
        plot_heatmap(model, test_loader, device, session_dir)
        # --- 指標テキスト保存 ---
        with open(os.path.join(session_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Mean Error: {mean_err:.4f}\n")
            f.write(f"Max Error: {max_err:.4f}\n")
            f.write(f"Accuracy@5px: {acc5:.4f}\n")
            f.write(f"Accuracy@10px: {acc10:.4f}\n")
            f.write("\nMean Error by Point:\n")
            for lbl, val in zip(point_labels, point_means):
                f.write(f"{lbl}: {val:.4f}\n")
        # --- 指標棒グラフ ---
        plt.figure()
        plt.bar(['Mean', 'Max', 'Acc@5px', 'Acc@10px'], [mean_err, max_err, acc5, acc10])
        plt.ylabel('Value')
        plt.title('Test Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, 'metrics_bar.png'))
        plt.close()
        writer.close()

    elif mode == '2':
        # 2番: predict+heatmap+onnx+特徴マップ
        imgp = input("画像ファイル名を相対パスで指定してください（例: img/0003.jpg または dataset/0023.jpg）: ").strip()
        # パス解釈
        if imgp.startswith('img/') or imgp.startswith('dataset/'):
            img_path = os.path.join(PARENT_DIR, imgp)
            folder_type = 'img' if imgp.startswith('img/') else 'dataset'
        else:
            # 旧仕様: img/ または dataset/ を自動補完
            img_path = None
            folder_type = None
            for folder, ftype in [(IMG_DIR, 'img'), (DATASET_DIR, 'dataset')]:
                candidate = os.path.join(folder, imgp)
                if os.path.exists(candidate):
                    img_path = candidate
                    folder_type = ftype
                    break
        if img_path is None or not os.path.exists(img_path):
            print(f"画像が見つかりません: {imgp}")
            return
        # out_dir = OUT_DIR
        out_dir = session_dir  # ← predict_with_featuresの出力先をセッションごとに
        model.load_state_dict(torch.load(PRED_CKPT, map_location=device, weights_only=True))
        pts, out_img, heatmap_path, fmap_dir, onnx_path = predict_with_features(model, img_path, device, out_dir)
        print(f"Predicted corners (1, 2, 3, 4): {pts}")
        print(f"Output image saved to: {out_img}")
        # heatmapはdataset/のときのみ出力
        if folder_type == 'dataset' and heatmap_path:
            model.load_state_dict(torch.load(PRED_CKPT, map_location=device,weights_only=True))
            success = plot_heatmap_for_image(model, img_path, device, heatmap_path)
            if success:
                print(f"ヒートマップを {heatmap_path} に保存しました。")
            else:
                print(f"heat_map error")
        elif folder_type == 'img':
            print("img/ フォルダの画像ではヒートマップは出力しません。")
        print(f"Feature maps saved to: {fmap_dir}")

    elif mode == '3':
        # imgフォルダ内の全画像を一括推論
        ckpt = PRED_CKPT
        model.load_state_dict(torch.load(ckpt, map_location=device,weights_only=True))
        model.eval()
        img_paths = sorted(glob(os.path.join(IMG_DIR, '*.jpg')))
        results = []
        for imgp in img_paths:
            pts, out_path = predict_and_plot(model, imgp, device)
            print(f"{os.path.basename(imgp)}: {pts} -> {os.path.basename(out_path)}")
            results.append((os.path.basename(imgp), pts, os.path.basename(out_path)))
        # 結果をテキストで保存
        result_txt = os.path.join(session_dir, 'batch_predict_results.txt')
        with open(result_txt, 'w', encoding='utf-8') as f:
            for name, pts, outimg in results:
                f.write(f"{name}: {pts} -> {outimg}\n")
        print(f"全画像の推論が完了しました。結果は {result_txt} に保存されました。")

    elif mode == '4':
        # 4番: カメラライブ推論
        model.load_state_dict(torch.load(PRED_CKPT, map_location=device, weights_only=True))
        from utils import predict_from_camera
        predict_from_camera(model, device, OUT_DIR)

    elif mode == '5':
        # 5番: ONNXエクスポート
        onnx_path = os.path.join(OUT_DIR, 'model.onnx')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(PRED_CKPT, map_location=device,weights_only=True))
        model.eval()
        dummy_input = torch.randn(1, 1, INPUT_SIZE[0], INPUT_SIZE[1], device=device)
        torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)
        print(f"ONNXファイルを {onnx_path} に保存しました。")



    else:
        print("Invalid mode selected. Exiting.")

#?----------------------------------------------------------------------------------------
#?ログ設定
#?----------------------------------------------------------------------------------------
os.makedirs('log/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('log/logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()

#? =========================================================================================
#? 各モードで実行される関数の流れ
#? =========================================================================================
#
#? 【学習機能（mode==1）】
# main
#  └─ for ep in range(...):#!===ループ===!#
#      ├─ train_one_epoch         # 1エポック分の学習を行う
#      ├─ validate               # 検証データで損失を計算
#      ├─ (if ep % DIST_THRESH == 0)
#      │    ├─ mean_error             # 平均誤差を算出
#      │    ├─ max_error              # 最大誤差を算出
#      │    ├─ accuracy_at_threshold  # 指定閾値内の正解率を算出
#      │    ├─ plot_heatmap           # 誤差のヒートマップを描画・保存
#      │    └─ (グラフ/テキスト出力など)
#      └─ (最後に再度 mean_error, max_error, accuracy_at_threshold, plot_heatmap)
#
#? 【predict機能（mode==2）】
# main
#  └─ 入力画像パス取得                # 推論対象画像のパスを取得
#      ├─ model.load_state_dict       # 学習済みモデルの重みを読み込み
#      ├─ predict_with_features       # 画像推論＋特徴マップ・ONNX・出力画像生成
#      ├─ (dataset画像の場合) plot_heatmap_for_image  # 画像1枚分の誤差ヒートマップを描画
#      └─ (結果出力/print)            # 推論結果やパスを出力
#
#
#? 【mode==3: batch_predict（一括推論）】
# main
#  ├─ model.load_state_dict           # 学習済みモデルの重みを読み込み
#  ├─ for imgp in img_paths:
#  │    └─ predict_and_plot           # 画像ごとに推論＋出力画像生成
#  └─ (結果をテキスト保存)            # 推論結果をテキストファイルに保存
#
#? 【export_onnx機能（mode==4）】
# main
#  ├─ model.load_state_dict           # 学習済みモデルの重みを読み込む
#  ├─ torch.onnx.export               # モデルをONNX形式でエクスポート
#  └─ (結果出力/print)                # エクスポート結果を出力
#

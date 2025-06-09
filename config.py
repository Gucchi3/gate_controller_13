import os

#共通設定値（パス、ハイパーパラメータ、カラーマップなど）
PARENT_DIR    = r'C:\Users\nnykb\Desktop\gate_controller'   # 親フォルダ
IMG_DIR       = os.path.join(PARENT_DIR, 'img')      # 推論用画像フォルダ
DATASET_DIR   = os.path.join(PARENT_DIR, 'dataset')  # 学習用画像フォルダ
JSON_DIR      = os.path.join(PARENT_DIR, 'json')     # <parent>/json
OUT_DIR       = os.path.join(PARENT_DIR, 'log')      # モデル・ログ保存先
IMG_OUT_DIR   = os.path.join(PARENT_DIR, 'img_out')  # 予測画像出力先
INPUT_SIZE    = (160, 160)      # 入力画像サイズ (H, W)
BATCH_SIZE    = 5               # バッチサイズ
LEARNING_RATE = 1e-4            # 学習率
EPOCHS        = 3000              # 学習エポック数
EVAL_PERIOD   = 50               # 何エポックごとに評価を行うか
NUM_WORKERS   = 4               # DataLoader workers
DIST_THRESH   = 5.0             #  
PRED_CKPT     = os.path.join(OUT_DIR, 'best.pth')  # predictで使うモデルパス

HEATMAP_CMAP = 'Reds'      # 点ごと誤差ヒートマップ用カラーマップ
HEATMAP_IMG_CMAP = 'Reds' # 画像ごと誤差ヒートマップ用カラーマップ
FEATUREMAP_CMAP = 'gray' # 特徴マップ可視化用カラーマップ
# グラフ色指定（mean, top, right, left, bottom）
MEAN_ERROR_CURVE_COLOR = "#5ea6ff"  # パステルブルー
POINT_ERROR_COLORS = ["#fd5162",  "#64f884",  "#fab662",  "#b964ff" ]
POINT_LABEL = ["1", "2", "3", "4"]
SHOW_SUMMAR = 1 #ネットワーク構造を出力するか

# augmentations全体のON/OFF
AUGMENTATION_ENABLED =True
# 画像変換パラメータ
FLIP_PROB = 0.5         # 左右反転の確率
ROTATE_PROB = 0       # ランダム回転の確率
ROTATE_DEGREE = 15      # 回転角度の最大値（±）
SCALE_PROB = 0.6        # 拡大縮小の確率
SCALE_RANGE = (0.4,1.2) # 拡大縮小の倍率範囲（min, max）
CONTRAST_PROB = 0.5     # コントラスト変換の確率
CONTRAST_RANGE = (0.6, 1.4) # コントラスト倍率範囲
BRIGHTNESS_PROB = 0.5   # 明るさ変換の確率
BRIGHTNESS_RANGE = (0.6, 1.4) # 明るさ倍率範囲
SHARPNESS_PROB = 0.5  # シャープネス変換の確率
SHARPNESS_RANGE = (0.6, 1.4) # シャープネス倍率範囲
# 入力画像保存のON/OFFと保存先ディレクトリ
SAVE_INPUT_IMG = True  # Trueで保存、Falseで保存しない
INPUT_IMG_DIR = r'log/input_img'  # 保存先ディレクトリ名
# 出力次元数（座標2×点数＋ゲート存在判定1）
NUM_OUTPUTS = len(POINT_LABEL)*2 + 1
# ゲート存在判定の損失重み
GATE_EXIST_LOSS_WEIGHT = 1.0  # 必要に応じて調整
# ゲート存在判定の閾値
GATE_EXIST_THRESH = 0.5
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
# Load CSVs
df_coords = pd.read_csv("original_1600_grid_points.csv")
df_valid = pd.read_csv("inside.csv", header=None, names=['X','Y'])
df_measurements = pd.read_csv("combined_sinr.csv")  # 30 × 7700
print(df_measurements.shape)


img = plt.imread("map_of_R1_Hall.png")
img_h, img_w = img.shape[0], img.shape[1]
print("Map size:", img_w, "x", img_h)

H, W = 40, 40

# Coordinates → Grid Indices
# Previous mapping func
def coords_to_indices(df, H, W):
    x_vals = df['X'].values
    y_vals = df['Y'].values

    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())

    x_idx = np.clip((x_norm * (W-1)).astype(int), 0, W-1)
    y_idx = np.clip((y_norm * (H-1)).astype(int), 0, H-1)

    return np.stack([y_idx, x_idx], axis=1)

# New mapping func
def coords_to_grid_binning(df, H=40, W=40, img_h=None, img_w=None):
    """
    df      : DataFrame (必须包含 X, Y 两列)
    H, W    : grid 尺寸
    img_h   : map_of_R1_Hall.png 高度
    img_w   : map_of_R1_Hall.png 宽度

    输出：grid_y, grid_x
    """

    # 自动提取
    xs = df["X"].values.astype(float)
    ys = df["Y"].values.astype(float)

    # 分桶映射（按比例，不压缩形状）
    x_idx = (xs / img_w * W).astype(int)
    y_idx = (ys / img_h * H).astype(int)

    # 边界保护
    x_idx = np.clip(x_idx, 0, W-1)
    y_idx = np.clip(y_idx, 0, H-1)

    return y_idx, x_idx


# Valid region mask
# valid_indices = coords_to_indices(df_valid, H, W)
# df_valid → grid index. valid_y and valid_x are lists of index
valid_y, valid_x = coords_to_grid_binning(df_valid, H, W, img_h, img_w)
# print(type(valid_y), valid_y)
# print(type(valid_y), valid_y)

valid_mask = torch.zeros((H, W), dtype=torch.bool)
for y, x in zip(valid_y, valid_x):
    valid_mask[y, x] = 1


# For a point, if its 8 neighbors are all inside points, then the point is considered as an inside point
def fill_holes_8_neighbors(mask):

    H, W = mask.shape
    new_mask = mask.copy()

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if mask[y, x] == 0:

                region = mask[y-1:y+2, x-1:x+2]

                if np.sum(region) == 8:
                    new_mask[y, x] = 1

    return new_mask


valid_indices = fill_holes_8_neighbors(valid_mask.cpu().numpy())
# Show valid mask
plt.figure(figsize=(6,6))
plt.imshow(valid_indices, cmap='gray', origin='upper')
plt.title("Valid Mask")
plt.colorbar(label="Inside (1) / Outside (0)")
plt.savefig("xgboost/valid_mask.png", dpi=300, bbox_inches='tight')
# plt.show()
# measurement_indices = coords_to_indices(df_measurements[['X','Y']], H, W)
sinr_y, sinr_x = coords_to_grid_binning(df_measurements[['X','Y']], H, W, img_h, img_w)
measurement_indices = np.stack([sinr_y, sinr_x], axis=1)

# print(measurement_indices.shape)
# print(measurement_indices[:10])

# Create Valid Mask
valid_mask = torch.zeros((H, W), dtype=torch.bool)
valid_mask[valid_indices[:,0], valid_indices[:,1]] = True
valid_coords_mask = valid_mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
mask_2d = valid_coords_mask[0, 0].cpu().numpy()  # shape → (40,40), 0: outside; 1: inside

# Extract SINR samples (30 × 7749)
data_array = df_measurements.iloc[:, 2:].values.astype(np.float32) # (30, 7749)
num_sensors, num_samples = data_array.shape
sensor_mean_raw = data_array.mean(axis=1)   # shape (30,)

ys = measurement_indices[:, 0]
xs = measurement_indices[:, 1]

X_train = np.vstack([xs, ys]).T   # (x, y)
Y_train = sensor_mean_raw         # (sinr)

inside_map_real = np.full((H, W), np.nan, dtype=np.float32)
for i in range(len(ys)):
    y = ys[i]
    x = xs[i]
    inside_map_real[y, x] = sensor_mean_raw[i]

plt.figure(figsize=(6,6))
plt.imshow(inside_map_real, cmap='viridis', origin='upper')
plt.colorbar(label="Ground Truth SINR (dB)")
plt.title("Ground Truth SINR (30 Sensors)")
plt.savefig("xgboost/gt.png", dpi=300, bbox_inches='tight')
# plt.show()


# Model training
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=1.0,
    colsample_bytree=1.0,
)

model.fit(X_train, Y_train)
grid_xy = np.array([(x,y) for y in range(H) for x in range(W)])
sinr_pred = model.predict(grid_xy)
sinr_map = sinr_pred.reshape(H, W)

for (x, y), pred in zip(grid_xy, sinr_pred):
    print(f"({x}, {y}) → {pred:.3f} dB")


inside_map = np.full((H, W), np.nan, dtype=np.float32)
ys, xs = np.where(valid_indices == True)  # Inside points

for y, x in zip(ys, xs):
    inside_map[y, x] = sinr_map[y, x]

plt.figure(figsize=(6,6))
plt.imshow(inside_map, cmap='viridis', origin='upper')
plt.colorbar(label="Predicted SINR (dB)")
plt.title("XGBoost Predicted SINR")
plt.savefig("xgboost/XGBoost_SINR.png", dpi=300, bbox_inches='tight')
# plt.show()


# Calculate MSE and validate

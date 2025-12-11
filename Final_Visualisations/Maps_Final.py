import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
from scipy.ndimage import zoom

print("="*70)
print("SINR Prediction Visualization: All Models")
print("="*70)

# ========== STEP 1: Load data ==========
print("\n[1/8] Loading data...")
map_path = "Map_of_R1_Hall.png"
img = Image.open(map_path)
img_array = np.array(img)
img_h, img_w = img_array.shape[0], img_array.shape[1]
print(f"  ✓ Map size: {img_w} x {img_h} pixels")

# Define training and test points directly (in pixel coordinates)
train_points_pixel = np.array([
    [562, 918],
    [725, 891],
    [534, 838],
    [344, 731],
    [670, 705],
    [372, 705],
    [589, 625],
    [317, 572],
    [725, 492],
    [779, 492],
    [833, 466],
    [317, 439],
    [534, 412],
    [426, 412],
    [290, 386],
    [670, 359],
    [426, 333],
    [589, 306],
    [263, 253],
    [236, 146],
    [534, 120],
    [507, 731],
    [426, 918],
    [344, 173],
    [589, 545],
    [507, 226],
    [426, 120],
    [643, 811],
    [344, 838],
    [372, 253]
])


test_points_pixel = np.array([
    [372, 466], [670, 492], [507, 545], [589, 705], [426, 811],
    [507, 918], [290, 625], [290, 173], [507, 173], [426, 306]
])

train_points_pixel_neel = np.array([
    [568, 923],
    [712, 883],
    [544, 844],
    [352, 743],
    [676, 718],
    [372, 696],
    [599, 633],
    [321, 574],
    [724, 497],
    [782, 482],
    [838, 470],
    [326, 448],
    [535, 407],
    [427, 408],
    [282, 396],
    [679, 370],
    [436, 328],
    [577, 301],
    [257, 257],
    [227, 159],
    [539, 113]
])


print(f"  ✓ Training points: {len(train_points_pixel)}")
print(f"  ✓ Test points: {len(test_points_pixel)}")

H, W = 40, 40

# Extract pixel coordinates
train_pixel_x = train_points_pixel[:, 0]
train_pixel_y = train_points_pixel[:, 1]
test_pixel_x = test_points_pixel[:, 0]
test_pixel_y = test_points_pixel[:, 1]
train_pixel_x_neel = train_points_pixel_neel[:, 0]
train_pixel_y_neel = train_points_pixel_neel[:, 1]
# ========== STEP 2: Load prediction grids for all models ==========
print("\n[2/8] Loading prediction grids...")

grid_x_coords, grid_y_coords = np.meshgrid(np.arange(W), np.arange(H))

# ========== XGBoost Training (3-column format: grid_x, grid_y, predicted_sinr) ==========
try:
    df_xgb_train = pd.read_csv("xgboost_train_sinr.txt")
    # Already in 3-column format
    print(f"  ✓ Loaded XGBoost Training: {len(df_xgb_train)} predictions (3-column format)")
except FileNotFoundError:
    try:
        df_xgb_train = pd.read_csv("xgboost_sinr.txt")
        print(f"  ✓ Loaded XGBoost Training (alt): {len(df_xgb_train)} predictions")
    except:
        print("  ✗ XGBoost training data not found!")
        df_xgb_train = None

# ========== XGBoost Test (40x40 grid format) ==========
try:
    df_xgb_test = pd.read_csv("xgboost_results_test.csv", header=None).values
    df_xgb_test = np.where(df_xgb_test == 0, np.nan, df_xgb_test)
    df_xgb_test_formatted = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': df_xgb_test.flatten()
    }).dropna()
    print(f"  ✓ Loaded XGBoost Test: {len(df_xgb_test_formatted)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ xgboost_results_test.csv not found!")
    df_xgb_test_formatted = None

# ========== GPR Training (3-column format: grid_x, grid_y, predicted_sinr) ==========
try:
    df_gpr_train_formatted = pd.read_csv("gpr_train.csv")
    # Already in 3-column format
    print(f"  ✓ Loaded GPR Training: {len(df_gpr_train_formatted)} predictions (3-column format)")
except FileNotFoundError:
    try:
        df_gpr_train_formatted = pd.read_csv("gpr_prediction_grid.csv")
        print(f"  ✓ Loaded GPR Training (alt): {len(df_gpr_train_formatted)} predictions")
    except:
        print("  ✗ GPR training data not found!")
        df_gpr_train_formatted = None

# ========== GPR Test (40x40 grid format) ==========
try:
    df_gpr_test = pd.read_csv("GPR_test_results.csv", header=None).values
    df_gpr_test = np.where(df_gpr_test == 0, np.nan, df_gpr_test)
    df_gpr_test_formatted = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': df_gpr_test.flatten()
    }).dropna()
    print(f"  ✓ Loaded GPR Test: {len(df_gpr_test_formatted)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ GPR_test_results.csv not found!")
    df_gpr_test_formatted = None

# ========== CVAE Train Mean (40x40 grid format) ==========
try:
    cvae_train_mu = pd.read_csv("cvae_train_mu.csv", header=None).values
    cvae_train_mu = np.where(cvae_train_mu == 0, np.nan, cvae_train_mu)
    df_cvae_train_mu = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': cvae_train_mu.flatten()
    }).dropna()
    print(f"  ✓ Loaded CVAE Train Mean: {len(df_cvae_train_mu)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ cvae_train_mu.csv not found!")
    df_cvae_train_mu = None

# ========== CVAE Train Std (40x40 grid format) ==========
try:
    cvae_train_std = pd.read_csv("cvae_train_std.csv", header=None).values
    cvae_train_std = np.where(cvae_train_std == 0, np.nan, cvae_train_std)
    df_cvae_train_std = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': cvae_train_std.flatten()
    }).dropna()
    print(f"  ✓ Loaded CVAE Train Std: {len(df_cvae_train_std)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ cvae_train_std.csv not found!")
    df_cvae_train_std = None

# ========== CVAE Test Mean (40x40 grid format) ==========
try:
    cvae_test_mu = pd.read_csv("cvae_test_mu.csv", header=None).values
    cvae_test_mu = np.where(cvae_test_mu == 0, np.nan, cvae_test_mu)
    df_cvae_test_mu = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': cvae_test_mu.flatten()
    }).dropna()
    print(f"  ✓ Loaded CVAE Test Mean: {len(df_cvae_test_mu)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ cvae_test_mu.csv not found!")
    df_cvae_test_mu = None

# ========== CVAE Test Std (40x40 grid format) ==========
try:
    cvae_test_std = pd.read_csv("cvae_test_std.csv", header=None).values
    cvae_test_std = np.where(cvae_test_std == 0, np.nan, cvae_test_std)
    df_cvae_test_std = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': cvae_test_std.flatten()
    }).dropna()
    print(f"  ✓ Loaded CVAE Test Std: {len(df_cvae_test_std)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ cvae_test_std.csv not found!")
    df_cvae_test_std = None

# ========== NEEL Mean (40x40 grid format) ==========
try:
    neel_mu = pd.read_csv("mu_grid_neel.csv", header=None).values
    neel_mu = np.where(neel_mu == 0, np.nan, neel_mu)
    df_neel_mu = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': neel_mu.flatten()
    }).dropna()
    print(f"  ✓ Loaded NEEL Mean: {len(df_neel_mu)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ mu_grid_neel.csv not found!")
    df_neel_mu = None

# ========== NEEL Std (40x40 grid format) ==========
try:
    neel_std = pd.read_csv("std_grid_neel.csv", header=None).values
    neel_std = np.where(neel_std == 0, np.nan, neel_std)
    df_neel_std = pd.DataFrame({
        'grid_x': grid_x_coords.flatten(),
        'grid_y': grid_y_coords.flatten(),
        'predicted_sinr': neel_std.flatten()
    }).dropna()
    print(f"  ✓ Loaded NEEL Std: {len(df_neel_std)} predictions (40x40 format)")
except FileNotFoundError:
    print("  ✗ std_grid_neel.csv not found!")
    df_neel_std = None

# ========== STEP 3: Create mask for walkable areas ==========
print("\n[3/8] Creating walkable area mask...")
if len(img_array.shape) == 3:
    img_gray = np.mean(img_array, axis=2)
else:
    img_gray = img_array

walkable_mask = img_gray > 200
print(f"  ✓ Walkable area: {np.sum(walkable_mask)} pixels ({100*np.sum(walkable_mask)/(img_h*img_w):.1f}%)")

# ========== STEP 4: Generate 40x40 grid points map ==========
print("\n[4/8] Generating 40×40 grid reference map...")

# Calculate grid point positions in pixels
grid_points_x = []
grid_points_y = []
for i in range(H):
    for j in range(W):
        pixel_x = (j / (W - 1)) * (img_w - 1)
        pixel_y = (i / (H - 1)) * (img_h - 1)
        grid_points_x.append(pixel_x)
        grid_points_y.append(pixel_y)

fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
ax.imshow(img_array, extent=[0, img_w, img_h, 0], aspect='auto', alpha=1.0)

# Plot 40×40 grid as tiny dots
ax.scatter(grid_points_x, grid_points_y, 
           s=5, c='gray', alpha=0.3, label='40×40 Grid Points', zorder=5)

# Plot test points as X marks
ax.scatter(test_pixel_x, test_pixel_y, 
           s=150, c='red', marker='x', linewidths=2, 
           label='Test Points (n=10)', zorder=10)

ax.set_title("Test Locations", 
             fontsize=18, weight='bold', pad=20)
ax.set_xlabel("X Position (pixels)", fontsize=13)
ax.set_ylabel("Y Position (pixels)", fontsize=13)
ax.set_xlim(0, img_w)
ax.set_ylim(img_h, 0)
ax.legend(loc='upper right', fontsize=12)
ax.grid(False)

plt.savefig("test_map.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✓ Saved: test_map.png")
plt.close()

# ========== STEP 5: Convert grid coordinates to pixel coordinates ==========
print("\n[5/8] Converting grid to pixel coordinates...")

def add_pixel_coords(df):
    if df is not None and isinstance(df, pd.DataFrame):
        if 'grid_x' in df.columns and 'pixel_x' not in df.columns:
            df['pixel_x'] = (df['grid_x'] / (W - 1)) * (img_w - 1)
            df['pixel_y'] = (df['grid_y'] / (H - 1)) * (img_h - 1)
    return df

df_xgb_train = add_pixel_coords(df_xgb_train)
df_xgb_test_formatted = add_pixel_coords(df_xgb_test_formatted)
df_gpr_train_formatted = add_pixel_coords(df_gpr_train_formatted)
df_gpr_test_formatted = add_pixel_coords(df_gpr_test_formatted)
df_cvae_train_mu = add_pixel_coords(df_cvae_train_mu)
df_cvae_train_std = add_pixel_coords(df_cvae_train_std)
df_cvae_test_mu = add_pixel_coords(df_cvae_test_mu)
df_cvae_test_std = add_pixel_coords(df_cvae_test_std)
df_neel_mu = add_pixel_coords(df_neel_mu)
df_neel_std = add_pixel_coords(df_neel_std)

print("  ✓ Pixel coordinates added")

# ========== STEP 6: Create interpolation function ==========
print("\n[6/8] Creating interpolation function...")

def create_interpolated_map(df, img_w, img_h):
    
    if df is None:
        return None
    
    if not isinstance(df, pd.DataFrame):
        return None
    
    if 'pixel_x' not in df.columns or len(df) == 0:
        return None
    
    dense_x = np.linspace(0, img_w - 1, img_w * 2)
    dense_y = np.linspace(0, img_h - 1, img_h * 2)
    grid_x, grid_y = np.meshgrid(dense_x, dense_y)
    
    points = np.column_stack((df['pixel_x'].values, df['pixel_y'].values))
    values = df['predicted_sinr'].values
    
    sinr_interpolated = griddata(
        points, 
        values, 
        (grid_x, grid_y), 
        method='cubic',
        fill_value=np.nan
    )
    return sinr_interpolated

# Create interpolated maps
print("  Creating interpolated maps...")
sinr_xgb_train = create_interpolated_map(df_xgb_train, img_w, img_h)
sinr_xgb_test = create_interpolated_map(df_xgb_test_formatted, img_w, img_h)
sinr_gpr_train = create_interpolated_map(df_gpr_train_formatted, img_w, img_h)
sinr_gpr_test = create_interpolated_map(df_gpr_test_formatted, img_w, img_h)
sinr_cvae_train_mu = create_interpolated_map(df_cvae_train_mu, img_w, img_h)
sinr_cvae_train_std = create_interpolated_map(df_cvae_train_std, img_w, img_h)
sinr_cvae_test_mu = create_interpolated_map(df_cvae_test_mu, img_w, img_h)
sinr_cvae_test_std = create_interpolated_map(df_cvae_test_std, img_w, img_h)
sinr_neel_mu = create_interpolated_map(df_neel_mu, img_w, img_h)
sinr_neel_std = create_interpolated_map(df_neel_std, img_w, img_h)

print("  ✓ Interpolation complete")

# ========== STEP 7: Apply walkable area mask ==========
print("\n[7/8] Applying walkable area mask...")

def apply_mask(sinr_map, walkable_mask):
    if sinr_map is None:
        return None
    zoom_factor_y = sinr_map.shape[0] / walkable_mask.shape[0]
    zoom_factor_x = sinr_map.shape[1] / walkable_mask.shape[1]
    walkable_mask_resized = zoom(walkable_mask.astype(float), (zoom_factor_y, zoom_factor_x), order=1) > 0.5
    return np.where(walkable_mask_resized, sinr_map, np.nan)

sinr_xgb_train_masked = apply_mask(sinr_xgb_train, walkable_mask)
sinr_xgb_test_masked = apply_mask(sinr_xgb_test, walkable_mask)
sinr_gpr_train_masked = apply_mask(sinr_gpr_train, walkable_mask)
sinr_gpr_test_masked = apply_mask(sinr_gpr_test, walkable_mask)
sinr_cvae_train_mu_masked = apply_mask(sinr_cvae_train_mu, walkable_mask)
sinr_cvae_train_std_masked = apply_mask(sinr_cvae_train_std, walkable_mask)
sinr_cvae_test_mu_masked = apply_mask(sinr_cvae_test_mu, walkable_mask)
sinr_cvae_test_std_masked = apply_mask(sinr_cvae_test_std, walkable_mask)
sinr_neel_mu_masked = apply_mask(sinr_neel_mu, walkable_mask)
sinr_neel_std_masked = apply_mask(sinr_neel_std, walkable_mask)

print("  ✓ Masks applied")

# ========== STEP 8: Visualization function ==========
print("\n[8/8] Generating visualizations...")

def create_visualization(sinr_masked, title, filename, points_x, points_y, point_label, vmin=None, vmax=None, cmap=None):
    """Create visualization with specified measurement points"""
    if sinr_masked is None:
        print(f"  ⊘ Skipped: {filename} (no data)")
        return
 # If vmin or vmax are not provided, compute defaults
    vmin = np.nanmin(sinr_masked) if vmin is None else vmin
    vmax = np.nanmax(sinr_masked) if vmax is None else vmax
    cmap='viridis' if cmap is None else cmap
    
    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
    
    ax.imshow(img_array, extent=[0, img_w, img_h, 0], aspect='auto', alpha=1.0)
    
    im = ax.imshow(
        sinr_masked, 
        cmap=cmap,
        origin='upper',
        extent=[0, img_w, img_h, 0],
        alpha=0.85,
        interpolation='hamming',
        vmin=vmin,
        vmax=vmax
    )
    
    # Add measurement points as tiny circles
    ax.scatter(points_x, points_y, 
               s=80, facecolors='none', edgecolors='red', linewidths=2, 
               label=point_label, zorder=10)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted SINR (dB)", fontsize=13, weight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    ax.set_xlabel("X Position (pixels)", fontsize=13)
    ax.set_ylabel("Y Position (pixels)", fontsize=13)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(False)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {filename}")
    plt.close()

# Generate all visualizations
"""
# XGBoost - Prediction Grid (30 training points)
create_visualization(sinr_xgb_train_masked, 
                    "XGBoost - Prediction Grid",
                    "XGBoost_train.png",
                    train_pixel_x, train_pixel_y, 
                    f"Training Points (n=30)")

# XGBoost - Test Grid (10 test points)
create_visualization(sinr_xgb_test_masked, 
                    "XGBoost - Test Grid",
                    "XGBoost_test.png",
                    test_pixel_x, test_pixel_y, 
                    f"Test Points (n=10)")
                    """
# CVAE Mean - Test Grid (10 test points)
create_visualization(sinr_cvae_test_mu_masked, 
                    "CVAE Mean - Test Grid",
                    "CVAE_Mean_test.png",
                    test_pixel_x, test_pixel_y, 
                    f"Test Points (n=10)", vmin=24, vmax=30)

# CVAE Std - Test Grid (10 test points)
create_visualization(sinr_cvae_test_std_masked, 
                    "CVAE Std - Test Grid",
                    "CVAE_Std_test.png",
                    test_pixel_x, test_pixel_y, 
                    f"Test Points (n=10)",vmin=0, vmax=1.8, cmap='magma' )


# CVAE Mean - Prediction Grid (30 training points)
create_visualization(sinr_cvae_train_mu_masked, 
                    "CVAE Mean - Prediction Grid",
                    "CVAE_Mean_train.png",
                    train_pixel_x, train_pixel_y, 
                    f"Training Points (n=30)",vmin=24, vmax=30)

# CVAE Std - Prediction Grid (30 training points)
create_visualization(sinr_cvae_train_std_masked, 
                    "CVAE Std - Prediction Grid",
                    "CVAE_Std_train.png",
                    train_pixel_x, train_pixel_y, 
                    f"Training Points (n=30)", cmap='magma')

# GPR - Prediction Grid (30 training points)
create_visualization(sinr_gpr_train_masked, 
                    "GPR - Prediction Grid",
                    "GPR_train.png",
                    train_pixel_x, train_pixel_y, 
                    f"Training Points (n=30)")

# GPR - Test Grid (10 test points)
create_visualization(sinr_gpr_test_masked, 
                    "GPR - Test Grid",
                    "GPR_test.png",
                    test_pixel_x, test_pixel_y, 
                    f"Test Points (n=10)")


# NEEL Mean - Prediction Grid (30 points)
create_visualization(sinr_neel_mu_masked, 
                    "NEEL Mean - Prediction Grid",
                    "NEEL_Mean_train.png",
                    train_pixel_x_neel, train_pixel_y_neel, 
                    f"Training Points ", vmin=16, vmax=32, cmap='YlOrRd')

# NEEL Std - Prediction Grid (30 points)
create_visualization(sinr_neel_std_masked, 
                    "NEEL Std - Prediction Grid",
                    "NEEL_Std_train.png",
                    train_pixel_x_neel, train_pixel_y_neel,
                    f"Training Points ", cmap='YlOrRd')

# ========== Summary ==========
print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
print("\nFiles created:")
print("  0. test_map.png - 40×40 grid reference + test locations")
print("\nPrediction Grid visualizations (30 training points):")
print("  1. XGBoost_train.png")
print("  2. GPR_train.png")
print("  3. CVAE_Mean_train.png")
print("  4. CVAE_Std_train.png")
print("  5. NEEL_Mean_train.png")
print("  6. NEEL_Std_train.png")
print("\nTest Grid visualizations (10 test points):")
print("  7. XGBoost_test.png")
print("  8. GPR_test.png")
print("  9. CVAE_Mean_test.png")
print("  10. CVAE_Std_test.png")
print(f"\nData statistics:")
print(f"  - Training points: {len(train_points_pixel)}")
print(f"  - Test points: {len(test_points_pixel)}")
print(f"  - Grid resolution: 40×40 = 1600 points")
print(f"  - Walkable area: {np.sum(walkable_mask)} pixels ({100*np.sum(walkable_mask)/(img_h*img_w):.1f}%)")

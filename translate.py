import pandas as pd
import os
import numpy as np

# 配置：全点文件 和 测量点文件 的路径（修改为你的文件名）
all_points_csv = "grid_coordinates.csv"         # 包含所有网格点的CSV（必须含 X,Y 列）
measured_points_csv = "measurement_to_grid.csv"    # 包含用于测量的点的CSV（必须含 X,Y 列）
output_flagged_csv = "points_with_measured_flag.csv"  # 输出文件
TOL_PIXELS = 1.0  # 匹配坐标时的容差（以像素为单位），可按需调整（例如 0.5, 1, 2）

def _find_col(df, candidates):
    """在DataFrame中查找首个存在于 candidates 的列名（大小写不敏感）"""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def mark_measured_points_by_coords(all_points_path, measured_points_path, out_path, tol=TOL_PIXELS):
    a = pd.read_csv(all_points_path)
    m = pd.read_csv(measured_points_path)

    # 识别坐标列
    x_col_all = _find_col(a, ['X', 'x', 'PosX', 'pos_x'])
    y_col_all = _find_col(a, ['Y', 'y', 'PosY', 'pos_y'])
    x_col_meas = _find_col(m, ['X', 'x', 'PosX', 'pos_x'])
    y_col_meas = _find_col(m, ['Y', 'y', 'PosY', 'pos_y'])

    if x_col_all is None or y_col_all is None:
        raise ValueError(f"全点文件无法识别 X/Y 列，请检查 {all_points_path} 的列名。")
    if x_col_meas is None or y_col_meas is None:
        raise ValueError(f"测量点文件无法识别 X/Y 列，请检查 {measured_points_path} 的列名。")

    # 转为 numpy array (float)
    a_coords = a[[x_col_all, y_col_all]].to_numpy(dtype=float)
    m_coords = m[[x_col_meas, y_col_meas]].to_numpy(dtype=float)

    # 对每个全点检查是否有测量点在 tol 范围内
    measured_flags = np.zeros(len(a_coords), dtype=int)
    if m_coords.size == 0:
        # 没有测量点
        measured_flags[:] = 0
    else:
        # 对于小数据集（如 1600 x 21），直接逐点计算距离足够快且简单
        for i, pt in enumerate(a_coords):
            d2 = ((m_coords - pt) ** 2).sum(axis=1)
            if np.any(d2 <= tol**2):
                measured_flags[i] = 1

    a['Measured'] = measured_flags

    # 输出文件，只保留 X, Y, Measured 三列
    out_df = a[[x_col_all, y_col_all, 'Measured']].copy()
    out_df.columns = ['X', 'Y', 'Measured']
    out_df.to_csv(out_path, index=False)
    print(f"Saved flagged points to: {out_path} (tol={tol} px)")

# 运行（仅当文件存在时才执行）
if os.path.exists(all_points_csv) and os.path.exists(measured_points_csv):
    try:
        mark_measured_points_by_coords(all_points_csv, measured_points_csv, output_flagged_csv, tol=TOL_PIXELS)
    except Exception as e:
        print(f"Error creating flagged CSV: {e}")
else:
    print("All-points CSV 或 measured-points CSV 不存在。请修改 translate.py 中的 all_points_csv 和 measured_points_csv 路径。")

import pandas as pd
import os
import numpy as np

# 配置：全点文件、测量点文件和 SINR 数据文件夹路径
measured_points_csv = "measurement_to_grid.csv"  # 包含测量点的CSV（必须含编号列 Point_Index）
sinr_data_folder = "sinr_data/params"            # SINR 数据文件夹路径
output_combined_csv = "measured_points_with_sinr.csv"  # 输出文件

def combine_measured_points_with_sinr(measured_points_path, sinr_folder, out_path):
    # 读取测量点
    measured_points = pd.read_csv(measured_points_path)
    
    # 识别列
    index_col = _find_col(measured_points, ['Point_Index', 'Index', 'ID', 'point_index'])
    x_col = _find_col(measured_points, ['X', 'x', 'PosX', 'pos_x'])
    y_col = _find_col(measured_points, ['Y', 'y', 'PosY', 'pos_y'])
    
    if index_col is None:
        raise ValueError(f"测量点文件无法识别编号列，请检查 {measured_points_path} 的列名。")
    if x_col is None or y_col is None:
        raise ValueError(f"测量点文件无法识别 X/Y 列，请检查 {measured_points_path} 的列名。")
    
    # 初始化结果列表
    combined_data = []
    
    # 遍历测量点
    for _, row in measured_points.iterrows():
        point_index = int(row[index_col])  # 点的编号
        x, y = row[x_col], row[y_col]
        
        # 构造对应的 SINR 文件名（假设文件名格式为 "radio_metrics_log<编号>.csv"）
        sinr_file = os.path.join(sinr_folder, f"radio_metrics_log{point_index}.csv")
        
        if os.path.exists(sinr_file):
            # 读取 SINR 数据，仅提取 "sinr" 列
            try:
                sinr_df = pd.read_csv(sinr_file, usecols=["SINR_dB"])  # 只读取 "sinr" 列
                sinr_values = sinr_df["SINR_dB"].values.flatten()  # 将 SINR 数据展平为一维数组
            except Exception as e:
                print(f"Error reading SINR file {sinr_file}: {e}")
                sinr_values = [np.nan]
        else:
            # 如果 SINR 文件不存在，则填充 NaN
            print(f"Warning: SINR file not found for point {point_index}")
            sinr_values = [np.nan]
        
        # 将测量点和对应的 SINR 数据合并
        combined_row = [x, y] + sinr_values.tolist()
        combined_data.append(combined_row)
    
    # 确定最大列数（因为每个点的 SINR 数据可能不同）
    max_sinr_count = max(len(row) for row in combined_data) - 2  # 减去 x, y 列
    
    # 创建列名
    columns = ['X', 'Y'] + [f"SINR_{i+1}" for i in range(max_sinr_count)]
    
    # 创建 DataFrame
    combined_df = pd.DataFrame(combined_data, columns=columns)
    
    # 保存到 CSV
    combined_df.to_csv(out_path, index=False)
    print(f"Saved combined data to: {out_path}")

def _find_col(df, candidates):
    """在 DataFrame 中查找首个存在于 candidates 的列名（大小写不敏感）"""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# 运行（仅当文件存在时才执行）
if os.path.exists(measured_points_csv) and os.path.exists(sinr_data_folder):
    try:
        combine_measured_points_with_sinr(measured_points_csv, sinr_data_folder, output_combined_csv)
    except Exception as e:
        print(f"Error combining measured points with SINR data: {e}")
else:
    print("Measured points CSV 或 SINR 数据文件夹不存在。请检查路径。")

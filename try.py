import csv
import sys
import argparse
import numpy as np
import pandas as pd

def read_points(path):
    """从 CSV 文件中读取坐标点"""
    pts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过标题行
        for row in reader:
            if not row:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
                pts.append((x, y))
            except:
                continue
    return pts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mask', default='mask.csv', help='Mask file (csv)')
    p.add_argument('--snapped', default='snapped_measurements.csv', help='Selected points (csv)')
    p.add_argument('--preprocessed', default='preprocessed_data.csv', help='Preprocessed data (csv)')
    p.add_argument('--out', default='sinr_output.csv', help='Output SINR csv')
    args = p.parse_args()

    # 读取 mask.csv
    try:
        mask_df = pd.read_csv(args.mask)
        mask_df.columns = mask_df.columns.str.lower()
        mask_arr = mask_df[['x', 'y', 'mask']].values
        print(f"Mask shape: {mask_arr.shape}")
    except Exception as e:
        print(f"Error reading {args.mask}: {e}", file=sys.stderr)
        sys.exit(1)

    # 读取 snapped_measurements.csv
    try:
        snapped_df = pd.read_csv(args.snapped)
        snapped_df.columns = snapped_df.columns.str.lower()
        print(f"Snapped shape: {snapped_df.shape}")
    except Exception as e:
        print(f"Error reading {args.snapped}: {e}", file=sys.stderr)
        sys.exit(1)

    # 读取 preprocessed_data.csv
    try:
        preprocessed_df = pd.read_csv(args.preprocessed)
        print(f"Preprocessed shape: {preprocessed_df.shape}")
    except Exception as e:
        print(f"Error reading {args.preprocessed}: {e}", file=sys.stderr)
        sys.exit(1)

    # 创建 Measurement_Point -> SINR 列表的映射
    sinr_map = {}
    for _, row in preprocessed_df.iterrows():
        point_id = int(row['Measurement_Point'])
        sinr_value = float(row['SINR_dB'])
        
        if point_id not in sinr_map:
            sinr_map[point_id] = []
        sinr_map[point_id].append(sinr_value)
    
    print(f"Found SINR data for {len(sinr_map)} points")

    # 创建坐标到 ID 的映射
    coord_to_id = {}
    for idx, row in snapped_df.iterrows():
        x = row['x'] if 'x' in snapped_df.columns else row['X']
        y = row['y'] if 'y' in snapped_df.columns else row['Y']
        coord_to_id[(x, y)] = idx + 1

    # 创建输出数据
    output_data = []
    max_sinr_count = 0

    # 第一遍：找到最大的 SINR 数据个数
    for x, y, measured in mask_arr:
        if measured == 1:
            point_id = coord_to_id.get((x, y))
            if point_id and point_id in sinr_map:
                sinr_values = sinr_map[point_id]
                max_sinr_count = max(max_sinr_count, len(sinr_values))

    print(f"Max SINR count: {max_sinr_count}")

    # 第二遍：生成输出数据
    for x, y, measured in mask_arr:
        if measured == 1:
            # 查找对应的点 ID
            point_id = coord_to_id.get((x, y))
            
            if point_id and point_id in sinr_map:
                sinr_values = sinr_map[point_id]
                row_data = [x, y] + sinr_values
            else:
                row_data = [x, y]
        else:
            # Measured == 0，SINR 数据都为 0
            row_data = [x, y]
        
        output_data.append(row_data)

    # 补齐所有行到相同长度，未测量的点填充 0
    for row in output_data:
        while len(row) < 2 + max_sinr_count:
            row.append(0)  # 所有未填充的位置都填 0

    print(f"Total output rows: {len(output_data)}")

    # 创建 DataFrame
    columns = ['X', 'Y'] + [f'SINR_{i+1}' for i in range(max_sinr_count)]
    output_df = pd.DataFrame(output_data, columns=columns)

    # 保存到 CSV
    output_df.to_csv(args.out, index=False)
    print(f"SINR data saved to {args.out}")

if __name__ == '__main__':
    main()



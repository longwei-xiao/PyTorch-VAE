import os
import pandas as pd

def main(combined_sinr_path, mask_path, output_file="combined_sinr_filled.csv"):
    print(f"正在读取 {combined_sinr_path}...")
    
    # 读取 combined_sinr.csv
    try:
        combined_df = pd.read_csv(combined_sinr_path)
        print(f"成功读取 combined_sinr.csv，共 {len(combined_df)} 行")
    except Exception as e:
        print(f"读取 combined_sinr.csv 失败: {e}")
        return
    
    # 读取 mask.csv
    print(f"正在读取 {mask_path}...")
    try:
        mask_df = pd.read_csv(mask_path)
        print(f"成功读取 mask.csv，共 {len(mask_df)} 行")
    except Exception as e:
        print(f"读取 mask.csv 失败: {e}")
        return
    
    # 确保列名一致（大小写不敏感）
    combined_df.columns = combined_df.columns.str.strip()
    mask_df.columns = mask_df.columns.str.strip()
    
    # 获取 SINR 列数
    sinr_columns = [col for col in combined_df.columns if col.startswith('SINR_')]
    num_sinr = len(sinr_columns)
    print(f"检测到 {num_sinr} 个 SINR 列")
    
    # 创建坐标到 SINR 数据的映射
    combined_set = set()
    sinr_map = {}
    
    for _, row in combined_df.iterrows():
        x, y = row['X'], row['Y']
        combined_set.add((x, y))
        sinr_values = row[sinr_columns].values.tolist()
        sinr_map[(x, y)] = sinr_values
    
    print(f"combined_sinr 中有 {len(combined_set)} 个坐标")
    
    # 收集所有 grid 中的坐标
    grid_set = set()
    grid_coords = []
    
    for _, row in mask_df.iterrows():
        x, y = row['x'], row['y']
        grid_set.add((x, y))
        grid_coords.append((x, y))
    
    print(f"mask 中有 {len(grid_set)} 个坐标")
    
    # 找出在 grid 中但不在 combined_sinr 中的坐标
    missing_coords = grid_set - combined_set
    print(f"需要添加 {len(missing_coords)} 个缺失的坐标")
    
    # 创建输出数据
    output_data = []
    
    # 先添加 combined_sinr 中已有的数据
    for _, row in combined_df.iterrows():
        output_data.append(row.values.tolist())
    
    # 再添加缺失的坐标（SINR 数据全部填充为 0）
    for x, y in sorted(missing_coords):
        row_data = [x, y] + [0] * num_sinr
        output_data.append(row_data)
    
    # 创建输出 DataFrame
    columns = ['X', 'Y'] + sinr_columns
    output_df = pd.DataFrame(output_data, columns=columns)
    
    # 按 X, Y 排序
    output_df = output_df.sort_values(['X', 'Y']).reset_index(drop=True)
    
    # 保存到 CSV
    output_path = output_file
    output_df.to_csv(output_path, index=False)
    
    print(f"\n成功！补齐后的文件已保存到: {output_path}")
    print(f"总共有 {len(output_df)} 行数据")
    print(f"其中 combined_sinr 中原有 {len(combined_df)} 行")
    print(f"新增 {len(missing_coords)} 行缺失坐标（SINR 数据为 0）")

if __name__ == "__main__":
    # 指定文件路径
    combined_sinr_path = "combined_sinr.csv"  # combined_sinr 文件路径
    mask_path = "mask.csv"  # mask 文件路径
    output_file = "combined_sinr_filled.csv"  # 输出文件路径
    
    main(combined_sinr_path, mask_path, output_file)
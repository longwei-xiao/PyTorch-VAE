import os
import glob
import pandas as pd

# preprepare.py
# 将指定目录下的所有 CSV 文件截断到相同的行数（以行数最少的文件为准）

def find_csv_files(directory, pattern="*.csv"):
    """查找指定目录下匹配模式的所有 CSV 文件"""
    path = os.path.join(directory, pattern)
    return sorted(glob.glob(path))

def main(target_directory):
    print(f"正在处理目录 {target_directory} 下的 CSV 文件...")
    
    # 查找指定目录下的所有 CSV 文件
    csv_files = find_csv_files(target_directory)
    if not csv_files:
        print(f"未找到任何 CSV 文件（目录：{target_directory}，模式：*.csv）")
        return

    # 读入并计算每个文件的行数（DataFrame 的记录数）
    dfs = {}
    for p in csv_files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print("读取失败：%s -> %s" % (p, e))
            return
        dfs[p] = df

    # 找到最少行数
    min_rows = min(len(df) for df in dfs.values())
    print("找到 %d 个 CSV，最少行数（数据行）为：%d" % (len(dfs), min_rows))

    # 截断并覆盖原文件
    for src_path, df in dfs.items():
        trimmed = df.iloc[:min_rows].copy()
        try:
            trimmed.to_csv(src_path, index=False)
            print("已截断并覆盖：%s（%d 行）" % (src_path, len(trimmed)))
        except Exception as e:
            print("写出失败：%s -> %s" % (src_path, e))

if __name__ == "__main__":
    # 指定目标目录
    target_directory = "sinr_data\params"  # 替换为你的目标目录路径
    main(target_directory)
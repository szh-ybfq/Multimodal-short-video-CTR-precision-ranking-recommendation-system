import pandas as pd
import os

data_dir = r"\MicroLens-100k\processed_final"  # 你的文件夹路径

# 2. 读取处理好的pkl文件
train_df = pd.read_parquet(data_dir + "\\train_final.parquet")
val_df = pd.read_parquet(data_dir + "\\val_final.parquet")
test_df = pd.read_parquet(data_dir + "\\test_final.parquet")
aux_data = pd.read_pickle(data_dir + "\\aux_data.pkl")

# 3. 查看数据基本信息（含前10行）
def inspect_data(df, name):
    print("="*60)
    print(f"【{name}】数据基本信息")
    print(f"总样本数：{len(df)}")
    print(f"列名：{df.columns.tolist()}")
    if "label" in df.columns:
        print(f"标签分布（1=正样本，0=负样本）：")
        print(df["label"].value_counts().to_string())
    print(f"\n{name} 前10行数据：")
    print(df.head(1000))
    print("="*60 + "\n")
# 分别查看三个数据集
inspect_data(train_df, "训练集")
inspect_data(val_df, "验证集")
inspect_data(test_df, "测试集")
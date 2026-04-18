import pandas as pd

# 读取你的 parquet 文件
df = pd.read_parquet("/Applications/Documents/RobustVQA/data/ScienceQA/train.parquet")

# 1. 查看前 5 行数据，看看长什么样
print("=== 前 5 行数据 ===")
print(df.head(2))

# 2. 查看这个表有哪些列、每列的数据类型
print("\n=== 数据集信息 ===")
print(df.info())

# 3. 如果某一行文本太长被折叠了，可以单独打印某一行某一列看看（比如第一行的 question）
print("\n=== 第一条题干内容 ===")
print(df.iloc[0]['question'])

import pandas as pd
import io
from PIL import Image

# 替换为你实际的文件路径
file_path = "/Applications/Documents/RobustVQA/data/ScienceQA/train.parquet" # 请改成你真实的路径
df = pd.read_parquet("/Applications/Documents/RobustVQA/data/ScienceQA")

# 我们直接取第 0 行数据（刚才看到这一行是有图片的）
sample = df.iloc[10].to_dict()

print("================ 完整单条样本展示 ================\n")

for key, value in sample.items():
    if key == 'image':
        if value is not None:
            # HuggingFace 的 Parquet 图片格式通常是一个包含 'bytes' 键的字典
            image_bytes = value['bytes']
            print(f"【{key}】: [包含真实图片流，大小为 {len(image_bytes)} 字节]")
            
            # 将二进制字节流转换为真实的图片并保存
            try:
                image = Image.open(io.BytesIO(image_bytes))
                save_path = "sample_image_0.png"
                image.save(save_path)
                print(f"  => 成功！已将该题干图片提取并保存为当前目录下的: {save_path} (你可以直接双击打开看图)")
            except Exception as e:
                print(f"  => 图片保存失败: {e}")
        else:
            print(f"【{key}】: None")
            
    elif isinstance(value, (list, tuple)):
        # 如果是选项列表，逐行打印更清晰
        print(f"【{key}】:")
        for i, item in enumerate(value):
            print(f"  {chr(65+i)}. {item}")
            
    else:
        # 其他文本类直接完整打印
        print(f"【{key}】: \n{value}")
        
    print("-" * 50)
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset

project_dir = os.path.abspath("")

shape = tuple(np.load(project_dir + '/preprocess_data/raw_blast/shape.npy'))
data = np.memmap(project_dir + '/preprocess_data/raw_blast/data.dat', dtype='float32', mode='r', shape=shape)

# 使用Parquet格式替代纯Arrow文件（更稳定）
output_dir = Path("temp_parquet_chunks")
output_dir.mkdir(exist_ok=True)

# 定义schema
schema = pa.schema([
    ('target', pa.list_(pa.float32(), 4096)),
    ('start', pa.timestamp('s')),
    ('freq', pa.string()),
    ('item_id', pa.string()),
])

batch_size = 19800  # 增大批次大小提升吞吐量
start_ts = pd.Timestamp('2023-01-01').timestamp()
freq_str = '1H'

# 分块写入Parquet文件
from tqdm import tqdm
for i in tqdm(range(0, data.shape[0], batch_size)):
    end_idx = min(i + batch_size, data.shape[0])
    # print(f"Processing batch {i//batch_size + 1}/{(shape[0]//batch_size)+1}")
    
    # 构建当前批次
    batch_target = data[i:end_idx].copy()
    flat_target = batch_target.reshape(-1).astype(np.float32)
    
    target_array = pa.FixedSizeListArray.from_arrays(flat_target, 4096)
    start_array = pa.array([start_ts] * (end_idx - i), pa.timestamp('s'))
    freq_array = pa.array([freq_str] * (end_idx - i), pa.string())
    item_id_array = pa.array([f'item_{j}' for j in range(i, end_idx)], pa.string())
    
    # 创建RecordBatch
    batch = pa.RecordBatch.from_arrays(
        [target_array, start_array, freq_array, item_id_array],
        schema.names
    )
    
    # 写入Parquet文件
    pq.write_table(
        pa.Table.from_batches([batch], schema=schema),
        output_dir / f"batch_{i//batch_size}.parquet"
    )

# 合并所有Parquet文件
dataset = Dataset.from_parquet(str(output_dir / "batch_*.parquet"))
dataset.save_to_disk(Path("example_dataset_1/blast"))

# 清理临时文件
import shutil
shutil.rmtree(output_dir)

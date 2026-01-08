import os
import torch
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, IndexedDataset

if __name__ == "__main__":
    indexed_dataset_path = "./"
    os.mkdir(indexed_dataset_path, exist_ok=True)
    builder = IndexedDatasetBuilder(os.path.join(indexed_dataset_path, "indexed_data.bin"))
    dataset = pq.ParquetDataset("/m2v_model/wuguohao03/dataset/github/dataset/data")
    tokenizer = AutoTokenizer.from_pretrained("/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen2.5-VL-7B-Instruct", use_fast=True)
    for f in dataset.fragments:
        batches = f.scanner(batch_size=128, columns=['content']).to_batches()
        for batch in batches:
            text_list = batch.column('content').tolist()
            input_ids_list = tokenizer(text_list, return_tensors='np')["input_ids"]
            sample_lengths = [len(input_ids) for input_ids in input_ids_list]
            builder.add_document(torch.from_numpy(np.concatenate(input_ids_list)), sample_lengths)
        builder.end_document()
    builder.finalize(indexed_dataset_path + "indexed_data.idx")


    # for test
    # ds = IndexedDataset(os.path.join(indexed_dataset_path, "indexed_data"))

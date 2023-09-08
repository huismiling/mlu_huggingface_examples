
import torch

from dataclasses import dataclass

@dataclass
class ModelConfig:
    id = ""
    use_int8 = True
    dataset = "samsum"
    max_source_length = 85
    max_target_length = 90
    output_dir = "model_outputs"
    saved_id = f"{output_dir}/saved_model/"
    save_dataset = True
    torch_dtype = torch.float32
    seed = None


use_int8 = True
t5_config = ModelConfig()
t5_config.id = "./flan-t5-xl/"
t5_config.use_int8 = use_int8
t5_config.saved_id = f"{t5_config.output_dir}/saved_model_{t5_config.torch_dtype}/"
if use_int8:
    t5_config.saved_id = f"{t5_config.output_dir}/saved_model_int8/"
    t5_config.torch_dtype = None
t5_config.seed = 7788
print(f"use_int8: {use_int8}, t5_config saved_id: {t5_config.saved_id}")


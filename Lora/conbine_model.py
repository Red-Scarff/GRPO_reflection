from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入分词器
from peft import PeftModel
import torch  # 导入 torch 库

# 定义路径
base_model_path = "/home/tione/notebook/Thinking_LLM/qwen/model/Qwen/Qwen2___5-Math-1___5B"
lora_path = "/home/tione/notebook/lora_output_openr1/final_adapter"
save_path = "/home/tione/notebook/Thinking_LLM/qwen/model/Qwen/Qwen2___5-Math-1___5B-Math-LoRA-Merged-v2"

# 加载原模型和分词器
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    # torch_dtype=torch.float16,
    # device_map="auto",
    # trust_remote_code=True  # Qwen 系列需要这个参数
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)  # 加载分词器

# 加载LoRA适配器
merged_model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = merged_model.merge_and_unload()  # 关键合并操作

# 保存完整模型和分词器
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)  # 保存分词器

print(f"融合后的模型已保存至：{save_path}")

























from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
import os

# 设置显存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    # 检查GPU状态
    assert torch.cuda.is_available(), "GPU不可用！"
    print(f"可用GPU数量：{torch.cuda.device_count()}，型号：{torch.cuda.get_device_name(0)}")

    # 加载数据集
    dataset = load_from_disk("/home/tione/notebook/Thinking_LLM/dataset/MATH-220k/Lora/processed_openr1_dataset")
    print(f"数据集加载成功，样本数：{len(dataset)}")

    # 加载分词器
    model_path = "/home/tione/notebook/Thinking_LLM/qwen/model/Qwen/Qwen2___5-Math-1___5B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"  # 确保与Flash Attention兼容
    )

    # 优化数据整理器
    def data_collator(features):
        # 提取文本内容
        texts = [f["text"] for f in features]
        
        # 编码文本
        batch = tokenizer(
            texts,
            padding="max_length",    # 固定长度填充
            truncation=True,
            max_length=2048,        # 根据实际数据长度调整
            return_tensors="pt"
        )
        
        # 添加labels字段
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # 模型加载优化
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",           # 自动多卡分配
        torch_dtype=torch.bfloat16,  # 使用bfloat16加载
        attn_implementation="flash_attention_2"  # 启用Flash Attention
    )

    # LoRA配置优化
    lora_config = LoraConfig(
        r=16,                        # 增加秩维度
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 覆盖更多层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"]  # 微调输出层
    )
    model = get_peft_model(model, lora_config)
    
    # 打印训练参数信息
    model.print_trainable_parameters()

    # 训练参数优化
    training_args = TrainingArguments(
        output_dir="./lora_output_openr1",
        per_device_train_batch_size=8,        # 降低batch_size
        gradient_accumulation_steps=8,        # 增加累积步数
        num_train_epochs=2,
        learning_rate=5e-4,                   # 提高学习率
        warmup_ratio=0.15,                    # 预热比例
        logging_steps=50,
        save_strategy="epoch",                # 每个epoch保存
        evaluation_strategy="no",
        bf16=True,                            # 启用bfloat16
        gradient_checkpointing=True,          # 显存优化
        optim="adamw_bnb_8bit",               # 8bit优化器
        report_to="none",
        save_total_limit=3,                   # 保留3个检查点
        lr_scheduler_type="cosine",
        group_by_length=False,                # 禁用长度分组
        remove_unused_columns=False,
        dataloader_num_workers=4,             # 减少数据加载进程
        dataloader_pin_memory=True           # 锁页内存加速
    )

    # 训练器配置
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 开始训练
    print("====== 开始训练 ======")
    trainer.train()
    print("====== 训练完成 ======")

    # 保存最终适配器
    final_save_path = os.path.join(training_args.output_dir, "final_adapter")
    model.save_pretrained(final_save_path)
    print(f"最终适配器已保存至：{final_save_path}")

except Exception as e:
    print(f"运行错误: {str(e)}")
    raise
finally:
    torch.cuda.empty_cache()
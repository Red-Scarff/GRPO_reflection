from modelscope import snapshot_download

# 指定模型名称和下载路径
model_name = "Qwen/Qwen2.5-Math-1.5B"
cache_dir = "/root/nfs/Thinking_LLM/qwen/model"  # 替换为你的目标路径

# 下载模型到指定路径
model_dir = snapshot_download(
    model_name, 
    cache_dir=cache_dir,  # 关键参数：指定下载路径
)

print(f"模型已下载到：{model_dir}")
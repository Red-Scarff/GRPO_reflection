from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"/root/nfs/Thinking_LLM/qwen/model/Qwen/Qwen2.5-1.5B-Instruct/"
# 使用本地模型的代码，注意修改model_path为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

history = []

while True:
    message = input('User: ')
    # 将用户输入加入对话历史
    history.append(message)
    
    # 将对话历史转换为模型输入
    input_text = ' '.join(history)
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    
    # 生成模型输出
    with torch.no_grad():
        outputs = model.generate(**inputs, num_return_sequences=1)
    
    # 解码生成的输出并提取文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 打印系统的响应
    print('System:', response)
    
    # 更新历史记录，保留最新输入的上下文
    history.append(response)

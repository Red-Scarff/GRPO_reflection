from modelscope import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\23605\.cache\huggingface\hub\models--Qwen--Qwen-1_8B-Chat\snapshots\1d0f68de57b88cfde81f3c3e537f24464d889081"
#使用本地模型的代码，注意修改model_path为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

history=None

while True:
    message = input('User:')
    response, history = model.chat(tokenizer, message, history=history)
    print('System:',end='')
    print(response)
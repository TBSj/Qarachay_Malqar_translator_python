# !pip -q install datasets loralib sentencepiece
# !pip -q install git+https://github.com/huggingface/transformers # need to install from github
# !pip -q install git+https://github.com/huggingface/peft.git
# !pip -q install bitsandbytes


from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

tokenizer = AutoTokenizer.from_pretrained("C:/Users/User/My_projects/Models/gpt-j-6B")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

from peft import PeftModel
from transformers import AutoTokenizer, GPTJForCausalLM, GenerationConfig, BitsAndBytesConfig

# device = "cuda" if torch.cuda.is_available() else "cpu"
# device

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

device_map = {
    "transformer.word_embeddings": "cpu",
    "transformer.word_embeddings_layernorm": "cpu",
    "lm_head": "cpu",
    "transformer.h": "cpu",
    "transformer.ln_f": "cpu",
    'transformer.wte.weight': "cpu"
}

model = GPTJForCausalLM.from_pretrained(
    "C:/Users/User/My_projects/Models/gpt-j-6B",
    # device_map=device_map,
    # quantization_config=quantization_config,
    # load_in_8bit=True,
    # # load_in_8bit_fp32_cpu_offload=True
    # device_map="auto"
) #.to('cpu')

model1 = PeftModel.from_pretrained(model, "samwit/dolly-lora", device_map=device_map,  quantization_config=quantization_config)



text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Что такое Цифровой двойник?
### Response:"""

def dolly_generate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cpu()

    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    print("Generating...")
    generation_output = model1.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
        pad_token_id = 0,
        eos_token_id = 50256
    )

    for s in generation_output.sequences:
        print(tokenizer.decode(s))
        


dolly_generate(text)

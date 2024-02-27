import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import pandas as pd
from datasets import Dataset, DatasetDict

#Lora Path
local_model_path = "Ubaidbhat/Finance" 

# # QLora..Preparing to load the Model in 4 bit...
bnb_config = BitsAndBytesConfig(

    #to quantize the model to 4-bits when you load it
    load_in_4bit = True,  
    
    #uses a nested quantization scheme to quantize the already quantized weights.
    bnb_4bit_use_double_quant = True, 
    
    #uses a special 4-bit data type for weights initialized from a normal distribution (NF4) #recommended from the paper...      
    bnb_4bit_quant_type = "nf4", 
    
    #While 4-bit bitsandbytes stores weights in 4-bits, the computation still happens in 16 or 32-bit
    bnb_4bit_compute_dtype=torch.bfloat16 
                     
)

#GPU mapping
d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None


# Loading the base Model
config = PeftConfig.from_pretrained(local_model_path)

print("loading base model.....................")
baseModel = AutoModelForCausalLM.from_pretrained(
  config.base_model_name_or_path,
  quantization_config = bnb_config,
  device_map=d_map,
  torch_dtype="auto",
  return_dict=True,
  trust_remote_code=True, 
  #use_flash_attention_2=True, 
  #ignore_mismatched_sizes=True 
    
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)



# load the base model with the Lora model
finetuneModel = PeftModel.from_pretrained(baseModel, local_model_path)




# merged = model.merge_and_unload()

prompt_template = """
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### Instruction:
  {query}

  ### Answer:
  """

prompt_template_with_context = """
  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that 
  appropriately completes the request.
  ### Instruction:
  {query}
  ### Input:
  {input}
  
  ### Answer:
  """



def extract_answer(message):
    # Find the index of '### Answer:'
    start_index = message.find('### Answer:')
    if start_index != -1:
        # Extract the part of the message after '### Answer:'
        answer_part = message[start_index + len('### Answer:'):].strip()
        # Find the index of the last full stop
        last_full_stop_index = answer_part.rfind('.')
        if last_full_stop_index != -1:
            # Remove the part after the last full stop
            answer_part = answer_part[:last_full_stop_index + 1]
        return answer_part.strip()  # Remove leading and trailing whitespace
    else:
        return "I don't have the answer to this question....."


#Inference Function............
def inferance(prompt: str, model, tokenizer, temp = 1.0, limit = 400, input = False) -> str:
    
  device = "cuda:0"
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to(device)

  generated_ids = model.generate(**model_inputs, max_new_tokens=limit, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
    
  answer  = extract_answer(decoded[0])
    
  return answer





load = pd.read_csv("eval.csv")
evalDataset = Dataset.from_pandas(load)






results = []
i = 0
p = False
for datapoint in evalDataset:
    print("Inference step {}.....................".format(i))
    i += 1
    question = datapoint["instruction"]
    correctAnswer = datapoint["output"]
    if datapoint['input']:
        prompt = prompt_template_with_context.format(query=question,input = datapoint['input'])
        baseModelAnswer = inferance(prompt, baseModel, tokenizer)
        fineTunedModelAnswer = inferance(prompt, finetuneModel, tokenizer)
    else:
        prompt = prompt_template.format(query=question)
        baseModelAnswer = inferance(prompt, baseModel, tokenizer)
        fineTunedModelAnswer = inferance(prompt, finetuneModel, tokenizer) 
    results.append({
        'Question': question,
        'correctAnswer': correctAnswer,
        'BaseModelAnswer': baseModelAnswer,
        'FineTunedModelAnswer': fineTunedModelAnswer
    })
    df = pd.DataFrame.from_dict(results)
    df.to_csv("benchDS.csv", index=False)
    























































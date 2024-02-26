# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel, PeftConfig
import pandas as pd
from datasets import Dataset, DatasetDict

# #Lora Path
# local_model_path = "outputs/checkpoint-4" 

# # QLora..Preparing to load the Model in 4 bit...
# bnb_config = BitsAndBytesConfig(

#     #to quantize the model to 4-bits when you load it
#     load_in_4bit = True,  
    
#     #uses a nested quantization scheme to quantize the already quantized weights.
#     bnb_4bit_use_double_quant = True, 
    
#     #uses a special 4-bit data type for weights initialized from a normal distribution (NF4) #recommended from the paper...      
#     bnb_4bit_quant_type = "nf4", 
    
#     #While 4-bit bitsandbytes stores weights in 4-bits, the computation still happens in 16 or 32-bit
#     bnb_4bit_compute_dtype=torch.bfloat16 
                     
# )

# #GPU mapping
# d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None


# # Loading the base Model
# config = PeftConfig.from_pretrained(local_model_path)

# print("-----------------------------loading base model-----------------------------------------------------------")
# baseModel = AutoModelForCausalLM.from_pretrained(
#   config.base_model_name_or_path,
#   quantization_config=bnb_config, 
#   device_map=d_map,  
#   use_cache=False, # set to False as we're going to use gradient checkpointing  
#   torch_dtype="auto",  
#   # ignore_mismatched_sizes=True  
#    trust_remote_code=True, 
#   #use_flash_attention_2=True,  
# )

# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)



# # load the base model with the Lora model
# finetuneModel = PeftModel.from_pretrained(baseModel, local_model_path)




# # merged = model.merge_and_unload()


# #Inference Function............
# def inferance(query: str, model, tokenizer, temp = 1.0, limit = 200) -> str:
#   device = "cuda:0"

#   prompt_template = """
#   Below is an instruction that describes a task. Write a response that appropriately completes the request.
#   ### Question:
#   {query}

#   ### Answer:
#   """
#   prompt = prompt_template.format(query=query)

#   encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

#   model_inputs = encodeds.to(device)

#   generated_ids = model.generate(**model_inputs, max_new_tokens=int(limit), temperature=temp, do_sample=True, 
#    pad_token_id=tokenizer.eos_token_id)
#   decoded = tokenizer.batch_decode(generated_ids)
#   return (decoded[0])

load = pd.read_csv("eval.csv")
evalDataset = Dataset.from_pandas(load)



results = []
i = 1
for datapoint in evalDataset:
    print("Inference step {}.....................".format(i))
    question = datapoint["instruction"]
    correctAnswer = datapoint["output"]
    # baseModelAnswer = inferance(question, baseModel, tokenizer)
    # fineTunedModelAnswer = inferance(question, finetuneModel, tokenizer)
    # Append results to the list
    results.append({
        'Question': question,
        'correctAnswer': correctAnswer,
        'BaseModelAnswer': "Helllooooooo",
        'FineTunedModelAnswer': "helooooooooooooooooooooooo"
    })
    i += 1
    if i == 4:
        break

df = pd.DataFrame.from_dict(results)
df.to_csv("benchDS.csv", index=False)




















































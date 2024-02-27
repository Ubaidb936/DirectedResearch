# -*- coding: utf-8 -*-
"""PreFinetuningForRunPod.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LtsUCcWfL2VpWLJXVkE5076XX5k3PTyg
"""

# IMPORTS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
from datasets import load_dataset
from trl import SFTTrainer
from peft import PeftConfig, PeftModel
from multiprocessing import cpu_count
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import pandas as pd
import transformers



# LOADING THE TOKENIZER
model_id = "mistralai/Mistral-7B-v0.1"
print("loading tokenizer..........................................................")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


print("Preparing dataset for fine-tuning..........................................")
# LOAD DATA FROM HUGGINFACE
data = load_dataset("gbharti/finance-alpaca", split = "train")


# PREPARE DATA FOR FINE-TUNING
def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    # Samples with additional context into.
    if data_point['input']:
        text = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Input:\n{data_point["input"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'

    # Without context
    else:
        text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
        text += f'### Instruction:\n{data_point["instruction"]}\n\n'
        text += f'### Response:\n{data_point["output"]}'
    return text

prompt = [generate_prompt(data_point) for data_point in data]
data = data.add_column("prompt", prompt);
data = data.map(lambda sample: tokenizer(sample["prompt"]),num_proc=cpu_count(), batched=True)
# data = data.remove_columns(['Context', 'Response'])
data = data.shuffle(seed=1234)
data = data.train_test_split(test_size=0.1)
train_data = data["train"]
test_data = data["test"]


df = pd.DataFrame(test_data)
df.to_csv('eval.csv', index=False)


# LOADING MODEL IN N(4, 8.....) BIT
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

d_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

print("loading model....................................")
model = AutoModelForCausalLM.from_pretrained(
model_id,
 torch_dtype="auto",
  use_cache=False, # set to False as we're going to use gradient checkpointing
  quantization_config=bnb_config,
  device_map=d_map,
  ignore_mismatched_sizes=True  
)


def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)


lora_config = LoraConfig(
    r=8,                                   # Number of quantization levels
    lora_alpha=32,                         # Hyperparameter for LoRA
    target_modules = modules, # Modules to apply LoRA to
    lora_dropout=0.05,                     # Dropout probability
    bias="none",                           # Type of bias
    task_type="CAUSAL_LM"                  # Task type (in this case, Causal Language Modeling)
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


# trainable, total = model.get_nb_trainable_parameters()
# print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()


MAXSTEPS = 100
NUM_WARMUP_STEPS = 30
SAVESTEPS = MAXSTEPS/4
EVALSTEPS = MAXSTEPS/2

training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        warmup_steps=NUM_WARMUP_STEPS,
        max_steps = MAXSTEPS,
        learning_rate=2e-4,
        output_dir="finance-outputs",
        optim="paged_adamw_8bit",
        evaluation_strategy = "steps",
        save_strategy="epoch",
        eval_steps = EVALSTEPS,
        save_steps = SAVESTEPS,
        logging_steps = 1,
        logging_dir = f"outputs/logs",
        lr_scheduler_type= "cosine",
        weight_decay=0.01,
        include_tokens_per_second=True,
    )


trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("finetuning starts........................")
model.config.use_cache = False
trainer.train()

model.push_to_hub("Finance")
tokenizer.push_to_hub("Finance")






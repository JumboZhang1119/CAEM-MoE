import warnings
import torch
import torch.nn as nn
import random
import os
import pickle
import re
import gc
import string
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import Dict, List
from torch.utils.data import DataLoader, Subset
from transformers import T5TokenizerFast, SwitchTransformersForConditionalGeneration, SwitchTransformersConfig
from datasets import load_from_disk
from rouge_score import rouge_scorer
from evaluate import load
squad_metric = load("squad")

# --- Environment and Model Configuration ---
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.set_device(0)

# --- Define storage paths ---
DATA_DIR = "/path/to/your/storage/Weight_File/"                 # Directory for agnews, squad, xsum datasets
SOURCE_MODEL_WEIGHT_PATH = "/path/to/your/storage/Weight_File/" # Path to load 3 single-task source model's weights
PICKLE_PATH = "/path/to/your/storage/Pickle_File/"              # Path to save pickle files (Training history)
MODEL_SAVE_PATH = "/path/to/your/storage/Weight_File/"          # Path to save CAEM multi-task model checkpoints

# --- Global Configurations ---
config = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "batch_size": 8,
    "lr": 3e-4,
    "num_epochs": 10,
    "gradient_accumulation_steps": 3,  # Effective batch size = batch_size * gradient_accumulation_steps
    "source_experts_num": 8,           # Number of experts in each source model
    "target_experts_num": 8,           # Number of experts in the final multi-task model
    "current_case_name": "CAEM",       # Name for the case
    "alpha": 1,                        # A parameter between 0 and 1 for the experts allocation ratio (0: uniform [CAEM-Avg], 1: skewed [CAEM])
}
task_array = [
    "ag_news",
    "squad",
    "xsum"
]

tokenizer = T5TokenizerFast.from_pretrained(f"google/switch-base-8")
file_name = f"{config['current_case_name']}_[{config['source_experts_num']}|{config['target_experts_num']}]"
print(file_name)

# --- Function to refresh and save history data ---
def refresh_saving_data(
    ag_news_new_data = None,
    squad_new_data = None,
    xsum_new_data = None,
    new_expert_usage_grid = None,
    new_expert_usage_grid_de = None,
    record_stay_num_experts_en = None,
    record_stay_num_experts_de = None
):
    with open(f"{PICKLE_PATH}{file_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    ag_news_data = loaded_data["ag_news_data"]
    squad_data = loaded_data["squad_data"]
    xsum_data = loaded_data["xsum_data"]
    expert_usage_grid = loaded_data["expert_usage_grid"]
    expert_usage_grid_de = loaded_data["expert_usage_grid_de"]
    experts_en = loaded_data["record_stay_num_experts_en"]
    experts_de = loaded_data["record_stay_num_experts_de"]
    if ag_news_new_data:
        ag_news_data.append(ag_news_new_data)
    if squad_new_data:
        squad_data.append(squad_new_data)
    if xsum_new_data:
        xsum_data.append(xsum_new_data)
    if new_expert_usage_grid:
        expert_usage_grid.append(new_expert_usage_grid)
    if new_expert_usage_grid_de:
        expert_usage_grid_de.append(new_expert_usage_grid_de)
    if record_stay_num_experts_en:
        experts_en = record_stay_num_experts_en
    if record_stay_num_experts_de:
        experts_de = record_stay_num_experts_de
    
    data = {
        "ag_news_data": ag_news_data,
        "squad_data": squad_data,
        "xsum_data": xsum_data,
        "expert_usage_grid": expert_usage_grid,
        "expert_usage_grid_de": expert_usage_grid_de,
        "record_stay_num_experts_en": experts_en,
        "record_stay_num_experts_de": experts_de
    }
        
    with open(f"{PICKLE_PATH}{file_name}.pkl", "wb") as f:
        pickle.dump(data, f)

ag_news_data = []
squad_data = []
xsum_data = []
expert_usage_grid = []
expert_usage_grid_de = []
record_stay_num_experts_en = []
record_stay_num_experts_de = []

data = {
    "ag_news_data": ag_news_data,
    "squad_data": squad_data,
    "xsum_data": xsum_data,
    "expert_usage_grid": expert_usage_grid,
    "expert_usage_grid_de": expert_usage_grid_de,
    "record_stay_num_experts_en": record_stay_num_experts_en,
    "record_stay_num_experts_de": record_stay_num_experts_de
}

os.makedirs(f"{PICKLE_PATH}{file_name}", exist_ok=True)
with open(f"{PICKLE_PATH}{file_name}.pkl", "wb") as f:
    pickle.dump(data, f)

# --- Load datasets ---
tokenized_ag_news = load_from_disk(f"{DATA_DIR}ag_news_dataset")
tokenized_squad = load_from_disk(f"{DATA_DIR}squad_dataset")
tokenized_xsum = load_from_disk(f"{DATA_DIR}xsum_dataset")
print(f"AG News: {len(tokenized_ag_news['train'])}")
print(f"SQuAD: {len(tokenized_squad['train'])}")
print(f"XSum: {len(tokenized_xsum['train'])}")

# --- CAEM Multi-Task Switch Transformer Model ---
class CAEM_MT_MoE(nn.Module):
    def __init__(
        self,
        task_type="ag_news",
        different_experts_merging = False,
    ):
        super().__init__()
        self.task_type = task_type
        if different_experts_merging:
            self.STmodel = SwitchTransformersForConditionalGeneration(SwitchTransformersConfig()).from_pretrained(f"google/switch-base-{config['target_experts_num']}")
        else:
            self.STmodel = SwitchTransformersForConditionalGeneration(SwitchTransformersConfig()).from_pretrained(f"google/switch-base-{config['source_experts_num']}")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.STmodel(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )
        return outputs
    
    def set_task_type(self, task_type):
        self.task_type = task_type
        
    @classmethod
    def CAEM(cls, models):
        merged_model = CAEM_MT_MoE(
            task_type = "ag_news",
            different_experts_merging = True,
        ).to(config['device'])
        merged_model.STmodel, record_stay_num_experts_en, record_stay_num_experts_de = merged_model.STmodel.caem(
            models,
            STmodel = merged_model.STmodel,
            en_num_experts = config['target_experts_num'],
            de_num_experts = config['target_experts_num'],
            num_layers = merged_model.STmodel.encoder.config.num_layers,
            alpha = config['alpha'],
        )
        # return the merged model and the experts allocation records
        return merged_model, record_stay_num_experts_en, record_stay_num_experts_de

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_dict = {key: [] for key in batch[0].keys()}
    for example in batch:
        for key, value in example.items():
            batch_dict[key].append(value)
    for key in batch_dict:
        if isinstance(batch_dict[key][0], (int, float)):
            batch_dict[key] = torch.tensor(batch_dict[key])
        elif isinstance(batch_dict[key][0], list):
            batch_dict[key] = torch.tensor(batch_dict[key])
    return batch_dict

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def testing(model = None, reset_flag = 1, use_val = None):
    test_model = model
    test_model.eval()
    all_predictions = []
    all_labels = []
    all_preds = []
    all_refs = []
    if (use_val):
        if (test_model.task_type == "ag_news"):
            test_data = tokenized_ag_news['validation']
        elif (test_model.task_type == "squad"):
            test_data = tokenized_squad['validation']
        elif (model.task_type == "xsum"):
            test_data = tokenized_xsum['validation']
    else:
        if (test_model.task_type == "ag_news"):
            test_data = tokenized_ag_news['test']
        elif (test_model.task_type == "squad"):
            test_data = tokenized_squad['test']
        elif (test_model.task_type == "xsum"):
            test_data = tokenized_xsum['test']

    test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=False)
    # --- Clear the expert usage counts ---
    if (reset_flag):
        for layer in range(1, test_model.STmodel.encoder.config.num_layers, test_model.STmodel.encoder.config.encoder_sparse_step): # every self.encoder.config.encoder_sparse_step layers have a expert layer
            test_model.STmodel.encoder.block[layer].layer[1].mlp.router.reset_expert_usage_counts()
            test_model.STmodel.decoder.block[layer].layer[2].mlp.router.reset_expert_usage_counts()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=80, dynamic_ncols=False):
            batch = {
                k: (v.to(config['device']) if hasattr(v, "to") else v)
                for k, v in batch.items()
            }
            generated_ids = test_model.STmodel.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=200,                 
                num_beams=3,                  
                repetition_penalty=1.2,     
                no_repeat_ngram_size=3,       
                length_penalty=1.0,           
                early_stopping=True,          
                temperature=1.0,             
            )
            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            decoded_preds = [pred.strip() for pred in decoded_preds]

            if (test_model.task_type == "squad"):
                for i, pred in enumerate(decoded_preds):
                    pred_id = str(batch["id"][i])
                    answer_texts = batch["answers"][i]["text"]
                    all_preds.append({
                        "id": pred_id,
                        "prediction_text": pred
                    })
                    all_refs.append({
                        "id": pred_id,
                        "answers": {
                            "text": answer_texts,
                            "answer_start": [0] * len(answer_texts)
                        }
                    })
            elif (test_model.task_type == "ag_news" or test_model.task_type == "xsum"):
                filtered_labels = [
                    [token for token in label if token != -100]
                    for label in batch['labels'].tolist()
                ]
                decoded_labels = tokenizer.batch_decode(
                    filtered_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                all_predictions.extend(decoded_preds)
                all_labels.extend(decoded_labels)

    # --- Record the expert usage counts ---
    emp_expert_grid = []
    emp_expert_grid_de = []
    for layer in range(1, test_model.STmodel.encoder.config.num_layers, test_model.STmodel.encoder.config.encoder_sparse_step): # every self.encoder.config.encoder_sparse_step layers have a expert layer
        usage_cnt = test_model.STmodel.encoder.block[layer].layer[1].mlp.router.get_expert_usage_counts()
        usage_cnt_de = test_model.STmodel.decoder.block[layer].layer[2].mlp.router.get_expert_usage_counts()
        if (reset_flag == 0 and test_model.task_type == task_array[-1] and use_val != 1):
            emp_expert_grid.append((usage_cnt/usage_cnt.sum().tolist()).tolist())
            emp_expert_grid_de.append((usage_cnt_de/usage_cnt_de.sum().tolist()).tolist())
    if (reset_flag == 0 and test_model.task_type == task_array[-1] and use_val != 1):
        refresh_saving_data(new_expert_usage_grid=emp_expert_grid, new_expert_usage_grid_de=emp_expert_grid_de)
        
    if test_model.task_type == "ag_news":
        exact_match_scores = []
        f1_scores = []
        for pred, label in zip(all_predictions, all_labels):
            norm_pred = normalize_answer(pred)
            norm_label = normalize_answer(label)

            is_exact_match = int(norm_pred == norm_label)
            exact_match_scores.append(is_exact_match)

            f1 = f1_score(pred, label)
            f1_scores.append(f1)
        exact_match = np.mean(exact_match_scores)
        f1 = np.mean(f1_scores)
        if (use_val):
            print('Task AG News [VALIDATION]: Acc: {:.7f}'.format(exact_match))
        else:
            print('Task AG News [TEST]: Acc: {:.7f}'.format(exact_match))
        if (use_val != 1):
            if (exact_match):
                refresh_saving_data(ag_news_new_data=exact_match)
            else:
                refresh_saving_data(ag_news_new_data=0.0001)
    
    elif test_model.task_type == "squad":
        results = squad_metric.compute(predictions=all_preds, references=all_refs)
        if (use_val):
            print('Task SQuAD [VALIDATION]: EM: {:.7f} | F1: {:.7f}'.format(results['exact_match'], results['f1']))
        else:
            print('Task SQuAD [TEST]: EM: {:.7f} | F1: {:.7f}'.format(results['exact_match'], results['f1']))
        if (use_val != 1):
            if (results['exact_match']):
                refresh_saving_data(squad_new_data=results['exact_match'])
            else:
                refresh_saving_data(squad_new_data=0.0001)
                
    elif test_model.task_type == "xsum": 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        scores = []
        for pred, label in zip(all_predictions, all_labels):
            score = scorer.score(pred, label)
            scores.append({
                'rouge1': score['rouge1'].fmeasure,
                'rouge2': score['rouge2'].fmeasure,
                'rougeL': score['rougeL'].fmeasure
            })
        avg_scores = {
            key: np.mean([s[key] for s in scores])
            for key in scores[0].keys()
        }
        if (use_val):
            print('Task XSum [VALIDATION]: ROUGE-1: {:.7f} | ROUGE-2: {:.7f} | ROUGE-L: {:.7f}'.format(
                avg_scores['rouge1'],
                avg_scores['rouge2'],
                avg_scores['rougeL']
            ))
        else:
            print('Task XSum [TEST]: ROUGE-1: {:.7f} | ROUGE-2: {:.7f} | ROUGE-L: {:.7f}'.format(
                avg_scores['rouge1'],
                avg_scores['rouge2'],
                avg_scores['rougeL']
            ))
        if (use_val != 1):
            if (avg_scores['rougeL']):
                refresh_saving_data(xsum_new_data=avg_scores['rougeL'])
            else:
                refresh_saving_data(xsum_new_data=0.0001)

# --- Mixed-Batch Dataset Rotator ---
class RotatingDataset:
    def __init__(self, base_dataset, epoch_length):
        self.base_dataset = base_dataset
        self.total_len = len(base_dataset)
        self.epoch_length = epoch_length
        self.current_start = 0

    def get_next_dataloader(self, batch_size, shuffle=True):
        end = self.current_start + self.epoch_length
        if end <= self.total_len:
            indices = list(range(self.current_start, end))
        else:
            indices = list(range(self.current_start, self.total_len)) + list(range(0, end % self.total_len))
        
        if shuffle:
            random.shuffle(indices)
            
        self.current_start = (self.current_start + self.epoch_length) % self.total_len

        subset = Subset(self.base_dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)

# --- Building all single-task models ---
model_1 = CAEM_MT_MoE().to(config['device'])
model_1.load_state_dict(torch.load(f"{SOURCE_MODEL_WEIGHT_PATH}single_task_{task_array[0]}.pt", map_location=config['device'], weights_only=True))
model_1.set_task_type(task_array[0])
print("Testing Result: ")
testing(model_1, reset_flag=1, use_val=0)
print("Validation Result: ")
testing(model_1, reset_flag=1, use_val=1)
print("=============================================================================")
    
model_2 = CAEM_MT_MoE().to(config['device'])
model_2.load_state_dict(torch.load(f"{SOURCE_MODEL_WEIGHT_PATH}single_task_{task_array[1]}.pt", map_location=config['device'], weights_only=True))
model_2.set_task_type(task_array[1])
print("Testing Result: ")
testing(model_2, reset_flag=1, use_val=0)
print("Validation Result: ") 
testing(model_2, reset_flag=1, use_val=1)
print("=============================================================================")

model_3 = CAEM_MT_MoE().to(config['device'])
model_3.load_state_dict(torch.load(f"{SOURCE_MODEL_WEIGHT_PATH}single_task_{task_array[2]}.pt", map_location=config['device'], weights_only=True))
model_3.set_task_type(task_array[2])
print("Testing Result:")
testing(model_3, reset_flag=1, use_val=0)
print("Validation Result:")
testing(model_3, reset_flag=1, use_val=1)
print("=============================================================================")
models_to_merge = [model_1, model_2, model_3]

# --- Merging all single-task models into a multi-task model ---
caem_model, record_stay_num_experts_en, record_stay_num_experts_de = CAEM_MT_MoE.CAEM(models = models_to_merge)
refresh_saving_data(record_stay_num_experts_en=record_stay_num_experts_en, record_stay_num_experts_de=record_stay_num_experts_de)
# --- Clean up memory ---
model_1.to("cpu")
del model_1
model_2.to("cpu")
del model_2
model_3.to("cpu")
del model_3
torch.cuda.empty_cache()
gc.collect()
caem_model.to(config['device'])
print("=============================================================================")

start_epoch = 0
epoch_length = len(tokenized_squad['train'])

ag_news_rotator = RotatingDataset(tokenized_ag_news['train'], epoch_length)
ag_news_rotator.current_start = (start_epoch * epoch_length) % len(tokenized_ag_news['train'])

squad_rotator = RotatingDataset(tokenized_squad['train'], epoch_length)
squad_rotator.current_start = (start_epoch * epoch_length) % len(tokenized_squad['train'])

xsum_rotator = RotatingDataset(tokenized_xsum['train'], epoch_length)
xsum_rotator.current_start = (start_epoch * epoch_length) % len(tokenized_xsum['train'])

# --- Fine-tuning the multi-task model ---
result = torch.cuda.get_device_name(config['device']) if torch.cuda.is_available() else "cpu"
print("Current device: " + result)
torch.cuda.empty_cache()
print(f"Current Encoder, Decoder Expert Numbers: {config['target_experts_num']}")
optimizer = torch.optim.AdamW(caem_model.parameters(), lr=config['lr'], eps=1e-6, betas=(0.9, 0.98), weight_decay=0.005)
for epoch in range(config['num_epochs']):
    ag_news_dl = ag_news_rotator.get_next_dataloader(batch_size=config['batch_size'])
    squad_dl = squad_rotator.get_next_dataloader(batch_size=config['batch_size'])
    xsum_dl = xsum_rotator.get_next_dataloader(batch_size=config['batch_size'])
    train_loss = 0
    caem_model.train()
    lrs = []
    optimizer.zero_grad()
    accumulation_steps = config["gradient_accumulation_steps"]
    total_steps = len(ag_news_dl)
    with tqdm(zip(ag_news_dl, squad_dl, xsum_dl), total=len(ag_news_dl), desc=f"Epoch {epoch + 1}/{config['num_epochs']}", ncols=100, dynamic_ncols=False) as progress_bar:
        for step, (batch_ag_news, batch_squad, batch_xsum) in enumerate(progress_bar):
            batches = [
                ('ag_news', batch_ag_news),
                ('squad', batch_squad),
                ('xsum', batch_xsum)
            ]
            for task_name, batch in batches:
                batch = {k: v.to(config['device']) for k, v in batch.items()}
                caem_model.set_task_type(task_name)
                    
                outputs = caem_model(
                    input_ids = batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss = loss / (accumulation_steps*3)
                loss.backward()
                train_loss += loss.item() * accumulation_steps
                lrs.append(optimizer.param_groups[0]['lr'])
                
            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
                optimizer.step()
                optimizer.zero_grad()
        train_loss = train_loss / (len(ag_news_dl))
        print('Epoch {:2d}: Learning Rate: {:.7f} Train Loss: {:.7f}'.format(epoch+1, lrs[-1], train_loss))
        torch.save(caem_model.state_dict(), f"{MODEL_SAVE_PATH}{file_name}_ep{epoch+1}.pt")
        print("==========================Testing==========================")
        for i in range(len(task_array)):
            caem_model.set_task_type(task_array[i])
            if (task_array[i] == "ag_news"):
                testing(caem_model, reset_flag = 1, use_val = 0)
            else:
                testing(caem_model, reset_flag = 0, use_val = 0)
        print("=========================Validation========================")
        for i in range(len(task_array)):
            caem_model.set_task_type(task_array[i])
            if (task_array[i] == "ag_news"):
                testing(caem_model, reset_flag = 1, use_val = 1)
            else:
                testing(caem_model, reset_flag = 0, use_val = 1)
        print("===========================================================")

# --- Show the final performance history and save the model ---
with open(f"{PICKLE_PATH}{file_name}.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"AG News Performance history: {loaded_data['ag_news_data']}")
print(f"SQuAD Performance history: {loaded_data['squad_data']}")
print(f"XSum Performance history: {loaded_data['xsum_data']}")
print(f"Encoder Experts allocation: {loaded_data['record_stay_num_experts_en']}")
print(f"Decoder Experts allocation: {loaded_data['record_stay_num_experts_de']}")
Merge_list = [(loaded_data['ag_news_data'][i]/loaded_data['ag_news_data'][0] + loaded_data['squad_data'][i]/loaded_data['squad_data'][0] + loaded_data['xsum_data'][i]/loaded_data['xsum_data'][0])/3 for i in range(len(loaded_data['ag_news_data']))]
best_merge = Merge_list.index(max(Merge_list[1:]))
print("Best Overall Epoch: ", best_merge)
print("AG News: ", loaded_data['ag_news_data'][best_merge])
print("SQuAD: ", loaded_data['squad_data'][best_merge])
print("XSum: ", loaded_data['xsum_data'][best_merge])

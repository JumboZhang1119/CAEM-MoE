import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, SwitchTransformersForConditionalGeneration, SwitchTransformersConfig
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
from typing import Dict, List
import os
import pickle
import re
import string
from evaluate import load
squad_metric = load("squad")

# --- Environment and Model Configuration ---
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.set_device(0)

# --- Define storage paths ---
DATA_DIR = "/path/to/your/storage/Weight_File/"           # Directory for agnews, squad, xsum datasets
PICKLE_PATH = "/path/to/your/storage/Pickle_File/"        # Path to save pickle files (Training history)
MODEL_SAVE_PATH = "/path/to/your/storage/Weight_File/"    # Path to save single-task model checkpoints

# --- Global Configurations ---
config = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "batch_size": 12,
    "lr": 3e-4,
    "num_epochs": 10,
    "max_input_length": 1000,
    "max_decoder_length": 200,
    "num_expert": 8,
    "gradient_accumulation_steps": 2,  # Effective batch size = batch_size * gradient_accumulation_steps
    "task_type": "ag_news",            # Specify the target task: "ag_news", "squad", or "xsum"
}

tokenizer = T5TokenizerFast.from_pretrained(f"google/switch-base-8")
file_name = f"single_task_{config['task_type']}"
PICKLE_PATH = f"{PICKLE_PATH}{file_name}.pkl"
MODEL_SAVE_PATH = f"{MODEL_SAVE_PATH}{file_name}.pt"

# --- Function to refresh and save history data ---
def refresh_saving_data(
    ag_news_new_data = None,
    squad_new_data = None,
    xsum_new_data = None
):
    with open(PICKLE_PATH, "rb") as f:
        loaded_data = pickle.load(f)
    ag_news_data = loaded_data["ag_news_data"]
    squad_data = loaded_data["squad_data"]
    xsum_data = loaded_data["xsum_data"]
    if ag_news_new_data:
        ag_news_data.append(ag_news_new_data)
    if squad_new_data:
        squad_data.append(squad_new_data)
    if xsum_new_data:
        xsum_data.append(xsum_new_data)
    data = {
        "ag_news_data": ag_news_data,
        "squad_data": squad_data,
        "xsum_data": xsum_data,
    }
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(data, f)

# --- CAEM Single-Task Switch Transformer Model ---
class CAEM_ST_MoE(nn.Module):
    def __init__(
        self,
        task_type="ag_news",
        different_experts_merging = False,
    ):
        super().__init__()
        self.task_type = task_type
        self.STmodel = SwitchTransformersForConditionalGeneration(SwitchTransformersConfig()).from_pretrained(f"google/switch-base-{config['num_expert']}")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.STmodel(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )
        return outputs
    
    def set_task_type(self, task_type):
        self.task_type = task_type

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

class Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, test_dataloader, task_type,
        learning_rate=config['lr'],
        num_epochs=config['num_epochs'],
        device=config['device']
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.task_type = task_type
        self.device = device
        learning_rate = config['lr']
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-6, betas=(0.9, 0.98), weight_decay=0.005)
        self.num_epochs = num_epochs
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        lrs = []
        self.optimizer.zero_grad()
        accumulation_steps = config["gradient_accumulation_steps"]
        total_steps = len(self.train_dataloader)
        for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training: Epoch {epoch + 1}/{config['num_epochs']}", ncols=100, dynamic_ncols=False)):   
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            remaining_steps = total_steps - step
            effective_accumulation = accumulation_steps if remaining_steps >= accumulation_steps else remaining_steps
            loss = outputs.loss
            loss = loss / effective_accumulation
            loss.backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss += loss.item() * effective_accumulation
            lrs.append(self.optimizer.param_groups[0]['lr'])
        return lrs, train_loss / len(self.train_dataloader)
    
    def evaluate(self, is_test=False):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_preds = []
        all_refs = []
        if is_test:
            dataloader = self.test_dataloader
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=config['device'], weights_only=True))
            self.model.eval()
        else:
            dataloader = self.val_dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ncols=70, dynamic_ncols=False):
                batch = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in batch.items()
                }
                generated_ids = self.model.STmodel.generate(
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
                decoded_preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                decoded_preds = [pred.strip() for pred in decoded_preds]

                if (self.task_type == "squad"):
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
                elif (self.task_type == "ag_news" or self.task_type == "xsum"):
                    filtered_labels = [
                        [token for token in label if token != -100]
                        for label in batch['labels'].tolist()
                    ]
                    decoded_labels = self.tokenizer.batch_decode(
                        filtered_labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    all_predictions.extend(decoded_preds)
                    all_labels.extend(decoded_labels)
        if self.task_type == "ag_news":
            metrics = self.compute_metrics(all_predictions, all_labels)
            return metrics
        elif self.task_type == "squad":
            results = squad_metric.compute(predictions=all_preds, references=all_refs)
            return {
                'accuracy': results['exact_match'],
                'f1': results['f1'],
            }
        else:
            metrics = self.compute_metrics(all_predictions, all_labels)
            return metrics
    
    def compute_metrics(self, predictions, labels):
        if self.task_type == "ag_news":
            exact_match_scores = []
            f1_scores = []
            for pred, label in zip(predictions, labels):
                norm_pred = normalize_answer(pred)
                norm_label = normalize_answer(label)

                is_exact_match = int(norm_pred == norm_label)
                exact_match_scores.append(is_exact_match)

                f1 = f1_score(pred, label)
                f1_scores.append(f1)
            exact_match = np.mean(exact_match_scores)
            f1 = np.mean(f1_scores)
            return {
                'accuracy': exact_match,
                'f1': f1
            }
            
        elif self.task_type == "xsum":  # seq2seq
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
                scores = []
                for pred, label in zip(predictions, labels):
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
                return {'accuracy': avg_scores}
            except ImportError:
                print("Warning: rouge_score not installed. Returning dummy scores.")
                return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

# --- Initialize Experiment Log File ---
with open(PICKLE_PATH, "wb") as f:
    pickle.dump({f"{config['task_type']}_data": []}, f)

def refresh_saving_data(new_metric_value):
    with open(PICKLE_PATH, "rb") as f:
        loaded_data = pickle.load(f)
    loaded_data[f"{config['task_type']}_data"].append(new_metric_value)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(loaded_data, f)

# --- Load Preprocessed Datasets ---
tokenized_datasets = {
    "ag_news": load_from_disk(f"{DATA_DIR}ag_news_dataset"),
    "squad": load_from_disk(f"{DATA_DIR}squad_dataset"),
    "xsum": load_from_disk(f"{DATA_DIR}xsum_dataset"),
}
dataset = tokenized_datasets[config['task_type']]

# --- Create DataLoaders ---
train_loader = DataLoader(dataset['train'], batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(dataset['validation'], batch_size=config['batch_size'], collate_fn=collate_fn)
test_loader = DataLoader(dataset['test'], batch_size=config['batch_size'], collate_fn=collate_fn)

# --- Initialize Model and Trainer ---
model = CAEM_ST_MoE().to(config['device'])

trainer = Trainer(
    model=model, tokenizer=tokenizer, train_dataloader=train_loader,
    val_dataloader=val_loader, test_dataloader=test_loader,
    task_type=config['task_type'], device=config['device']
)

# --- Training Loop ---
print(f"Starting single-task training for: {config['task_type']}")
valid_acc_max = 0.0

for epoch in range(config['num_epochs']):
    current_epoch = epoch + 1
    lr, train_loss = trainer.train_epoch(epoch)
    valid_result = trainer.evaluate()
    primary_metric = valid_result['accuracy']
    
    # Determine primary metric for comparison
    if (config['task_type'] == "ag_news"):
        print('Epoch {:2d}: Learning Rate: {:.7f} | Train Loss: {:.7f} | Val Acc: {:.7f} | Val F1: {:.7f}'.format(
                current_epoch,
                lr[-1],
                train_loss,
                valid_result['accuracy'],
                valid_result['f1']
            ))
    elif (config['task_type'] == "squad"):
        print('Epoch {:2d}: Learning Rate: {:.7f} | Train Loss: {:.7f} | Val EM: {:.7f} | Val F1: {:.7f}'.format(
                current_epoch,
                lr[-1],
                train_loss,
                valid_result['accuracy'],
                valid_result['f1']
            ))
    elif (config['task_type'] == "xsum"):
        print('Epoch {:2d}: Learning Rate: {:.7f} | Train Loss: {:.7f} | Val rouge1: {:.7f} | Val rouge2: {:.7f} | Val rougeL: {:.7f}'.format(
                current_epoch,
                lr[-1],
                train_loss,
                primary_metric['rouge1'],
                primary_metric['rouge2'],
                primary_metric['rougeL']
            ))
        primary_metric = primary_metric['rougeL']
    refresh_saving_data(primary_metric)
    if primary_metric >= valid_acc_max:
        print(f"Validation score improved ({valid_acc_max:.4f} --> {primary_metric:.4f}). Saving model...")
        valid_acc_max = primary_metric
        trainer.save_model(MODEL_SAVE_PATH)
with open(PICKLE_PATH, "rb") as f:
    loaded_data = pickle.load(f)
print(f"Training history for {config['task_type']}:")
print(loaded_data[f"{config['task_type']}_data"])
print("--- Training Complete ---")

# --- Final Evaluation on Test Set ---
print("Loading best model for final testing...")
test_result = trainer.evaluate(is_test=True)
print("Final Test Set Performance:")
primary_metric = test_result['accuracy']
if (config['task_type'] == "ag_news"):
    print('Test Acc: {:.7f} | Test F1: {:.7f}'.format(test_result['accuracy'], test_result['f1']))
elif (config['task_type'] == "squad"):
    print('Test EM: {:.7f} | Test F1: {:.7f}'.format(test_result['accuracy'], test_result['f1']))
elif (config['task_type'] == "xsum"):
    print('Test ROUGE-1: {:.7f} | Test ROUGE-2: {:.7f} | Test ROUGE-L: {:.7f}'.format(primary_metric['rouge1'], primary_metric['rouge2'], primary_metric['rougeL']))
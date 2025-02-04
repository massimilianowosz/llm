import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os
import re

from tokenizer import Tokenizer
from model import LLM
from dataset import WikiTextDataset, collate_fn
import config

# =============================================================================
# Configurazione della cartella dei checkpoint
# =============================================================================

def get_latest_checkpoint(checkpoint_dir):
    """
    Restituisce il percorso dell'ultimo checkpoint salvato (in base all'epoca) oppure None se non ve ne sono.
    """
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        return None
    # Ordina i file in base al numero di epoca (estratto dal nome)
    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1
    checkpoint_files = sorted(checkpoint_files, key=lambda f: extract_epoch(f))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

latest_checkpoint = get_latest_checkpoint(config.checkpoint_dir)
start_epoch = 0

# =============================================================================
# Inizializzazione del Tokenizer
# =============================================================================
tokenizer = Tokenizer(config.encoder_path, config.bpe_path)
vocab_size = len(tokenizer.encoder)

# =============================================================================
# Inizializzazione del modello (LLM) per il pre-training
# =============================================================================
model = LLM(vocab_size, config.d_model, config.nhead, config.num_layers, config.max_seq_len, config.dropout)
model = model.to(config.device)

# =============================================================================
# Caricamento del dataset per il pre-training
# =============================================================================
print("Caricamento del dataset WikiText per pre-training...")
dataset_wikitext = load_dataset("wikitext", config.pretrain_dataset, split="train")
texts = dataset_wikitext["text"]
dataset = WikiTextDataset(texts, tokenizer, config.max_seq_len)

# Preparazione del token di padding (usiamo il token EOS, rappresentato da una stringa vuota)
pad_token = ""
pad_token_id = tokenizer.encoder.get(pad_token, 0)

dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    pin_memory=True
)

# =============================================================================
# Configurazione dell'ottimizzatore e della funzione di loss
# =============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

# =============================================================================
# Caricamento del checkpoint (se esiste) per riprendere l'addestramento
# =============================================================================
if latest_checkpoint is not None:
    print(f"Checkpoint trovato: {latest_checkpoint}. Carico il checkpoint...")
    checkpoint = torch.load(latest_checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Riprendo dall'epoca successiva
    print(f"Riprendo l'addestramento dall'epoca {start_epoch}")
else:
    print("Nessun checkpoint trovato, inizio training da zero.")

# =============================================================================
# Training loop per il pre-training
# =============================================================================
model.train()
for epoch in range(start_epoch, config.num_epochs_pretrain):
    total_loss = 0.0
    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Pre-training Epoch {epoch+1}/{config.num_epochs_pretrain}"
    )
    
    for batch_idx, batch in progress_bar:
        batch = batch.to(config.device)
        seq_len = batch.size(1)
        
        # Creazione della maschera causale per l'attenzione (language modeling autoregressivo)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(config.device)
        
        optimizer.zero_grad()
        logits = model(batch, mask)
        
        # Preparazione dei dati per il calcolo della loss (shiftiamo le sequenze)
        logits = logits[:, :-1, :].reshape(-1, vocab_size)
        targets = batch[:, 1:].reshape(-1)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    print(f"Pre-training Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # Salvataggio del checkpoint al termine dell'epoca corrente
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        # Eventuali altri stati (es. scheduler) possono essere aggiunti qui
    }
    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint salvato: {checkpoint_path}")

# =============================================================================
# Salvataggio finale del modello pre-trained
# =============================================================================
torch.save(model.state_dict(), config.pretrained_model_path)
print(f"Modello pre-trained salvato in: {config.pretrained_model_path}")
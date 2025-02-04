import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Importiamo il Tokenizer (per la tokenizzazione dei prompt)
from tokenizer import Tokenizer
# Importiamo il modello LLM (definito in model.py)
from model import LLM
# Importiamo il dataset e la funzione collate (gestisce il padding dei batch)
from dataset import InstructDataset, collate_fn
# Importiamo la configurazione centralizzata
import config

# -----------------------------------------------------------------------------
# Inizializzazione del Tokenizer
# -----------------------------------------------------------------------------
# Il Tokenizer viene inizializzato con i file encoder.json e vocab.bpe definiti in config.py.
tokenizer = Tokenizer(config.encoder_path, config.bpe_path)
# Determiniamo la dimensione del vocabolario in base al dizionario del tokenizer.
vocab_size = len(tokenizer.encoder)

# -----------------------------------------------------------------------------
# Inizializzazione del modello per l'instruct-tuning
# -----------------------------------------------------------------------------
# Creiamo un'istanza del modello LLM (definito in model.py) usando i parametri dal file config.py.
model = LLM(vocab_size, config.d_model, config.nhead, config.num_layers, config.max_seq_len, config.dropout)
# Spostiamo il modello sul device configurato (su Mac M3, questo sarà "mps" se disponibile, altrimenti "cpu").
model = model.to(config.device)

# -----------------------------------------------------------------------------
# Caricamento del modello pre-trained
# -----------------------------------------------------------------------------
# Per l'instruct-tuning partiamo da un modello già pre-addestrato. I pesi pre-trained sono salvati nel percorso
# specificato in config.pretrained_model_path. Qui li carichiamo e li assegniamo al modello.
pretrained_state = torch.load(config.pretrained_model_path, map_location=config.device)
model.load_state_dict(pretrained_state)

# -----------------------------------------------------------------------------
# Caricamento del dataset per l'instruct-tuning (Alpaca)
# -----------------------------------------------------------------------------
print("Caricamento del dataset Alpaca per instruct-tuning...")
# Utilizziamo la libreria datasets per caricare il dataset "yahma/alpaca-cleaned" in modalità "train".
alpaca_dataset = load_dataset(config.instruct_dataset, split="train")
data = alpaca_dataset  # In questo esempio, usiamo direttamente il dataset caricato

# Creiamo un'istanza del dataset personalizzato per l'instruct-tuning. La classe InstructDataset si occupa di:
# - Costruire, per ogni esempio, un prompt formattato a partire dai campi "instruction", "input" e "output".
# - Tokenizzare il prompt e troncandolo se supera la lunghezza massima.
dataset = InstructDataset(data, tokenizer, config.max_seq_len)

# -----------------------------------------------------------------------------
# Configurazione del padding per il DataLoader
# -----------------------------------------------------------------------------
# Per il padding usiamo il token EOS, rappresentato qui dalla stringa vuota "".
pad_token = ""
# Otteniamo l'ID del token di padding dal vocabolario del tokenizer; se non esiste, usiamo 0 come fallback.
pad_token_id = tokenizer.encoder.get(pad_token, 0)

# Creiamo il DataLoader per iterare sul dataset. 
# - batch_size: numero di esempi per batch, definito in config.
# - shuffle=True: gli esempi vengono mescolati ad ogni epoca per migliorare l'apprendimento.
# - collate_fn: una funzione personalizzata che effettua il padding delle sequenze, in modo che tutte abbiano la stessa lunghezza.
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id)
)

# -----------------------------------------------------------------------------
# Configurazione dell'ottimizzatore e della funzione di loss
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
# La CrossEntropyLoss viene utilizzata per problemi di language modeling dove il modello deve predire il token successivo.

# -----------------------------------------------------------------------------
# Training loop per l'instruct-tuning
# -----------------------------------------------------------------------------
# Mettiamo il modello in modalità training.
model.train()
for epoch in range(config.num_epochs_instruct):
    total_loss = 0.0
    # Creiamo una barra di avanzamento con tqdm per monitorare il progresso durante l'epoca corrente.
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Instruct-tuning Epoch {epoch+1}/{config.num_epochs_instruct}")
    
    # Iteriamo su ogni batch nel DataLoader.
    for batch_idx, batch in progress_bar:
        # Spostiamo il batch sul device configurato.
        batch = batch.to(config.device)
        # Determiniamo la lunghezza della sequenza nel batch (tutti i batch hanno la stessa lunghezza dopo il padding).
        seq_len = batch.size(1)
        
        # Creiamo la maschera causale per il self-attention.
        # La maschera è una matrice triangolare superiore che impedisce al modello di "guardare" token futuri durante il training.
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(config.device)
        
        # Azzeriamo i gradienti dell'ottimizzatore per il batch corrente.
        optimizer.zero_grad()
        
        # Effettuiamo una forward pass del modello:
        # - Il batch (sequenze di token) e la maschera vengono passati al modello.
        logits = model(batch, mask)
        
        # Prepariamo i dati per il calcolo della loss:
        # - Per language modeling, l'input al modello è la sequenza troncata (tutti i token tranne l'ultimo)
        #   e il target è la stessa sequenza spostata di una posizione (tutti i token tranne il primo).
        logits = logits[:, :-1, :].reshape(-1, vocab_size)
        targets = batch[:, 1:].reshape(-1)
        
        # Calcoliamo la loss confrontando la previsione del modello (logits) con il target.
        loss = criterion(logits, targets)
        
        # Effettuiamo la backpropagation per calcolare i gradienti.
        loss.backward()
        # Aggiorniamo i pesi del modello utilizzando l'ottimizzatore (Adam).
        optimizer.step()
        
        # Accumuliamo la loss per il calcolo della loss media al termine dell'epoca.
        total_loss += loss.item()
        # Aggiorniamo la barra di avanzamento con il valore corrente della loss.
        progress_bar.set_postfix(loss=loss.item())
    
    # Calcoliamo la loss media per l'epoca corrente e la stampiamo.
    avg_loss = total_loss / len(dataloader)
    print(f"Instruct-tuning Epoch {epoch+1} Loss: {avg_loss:.4f}")

# -----------------------------------------------------------------------------
# Salvataggio del modello fine-tuned (instruct)
# -----------------------------------------------------------------------------
# Al termine del training, salviamo lo stato dei pesi del modello nel file definito in config.model_path.
torch.save(model.state_dict(), config.model_path)
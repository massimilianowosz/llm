import torch
import os
# -------------------------------
# Parametri del modello
# -------------------------------
d_model = 256  
# La dimensione degli embedding e delle rappresentazioni interne.
# Ogni token verrà trasformato in un vettore di 256 dimensioni.

nhead = 8  
# Il numero di "teste" per il meccanismo di multi-head attention.
# Più teste consentono al modello di catturare relazioni diverse nella sequenza.

num_layers = 4  
# Il numero di strati (o blocchi) Transformer che compongono il modello.
# Ogni layer aggiunge una maggiore capacità di apprendere rappresentazioni complesse.

max_seq_len = 256  
# La lunghezza massima delle sequenze di input (in token).
# Se una sequenza supera questo numero, verrà troncata per rientrare in questo limite.

dropout = 0.1  
# Il tasso di dropout usato per la regularizzazione durante il training.
# Durante il dropout, il 10% delle connessioni viene disabilitato per evitare l'overfitting.

# -------------------------------
# Parametri del training
# -------------------------------
learning_rate = 1e-4
# Il tasso di apprendimento usato dall'ottimizzatore per aggiornare i pesi.
# Un valore di 0.001 significa che ogni aggiornamento dei pesi avrà questa dimensione.

# Numero di epoche da eseguire nelle diverse fasi del training:
num_epochs_pretrain = 3  
# Numero di epoche per il pre‑training su un dataset non supervisionato (es. WikiText-103 o The Pile).
# In un contesto reale questo numero potrebbe essere molto più alto.

num_epochs_instruct = 3  
# Numero di epoche per il fine-tuning (instruct-tuning) sul dataset di istruzione–risposta (es. Alpaca).
# Anche qui il numero può variare a seconda del dataset e degli obiettivi.

batch_size = 2 
# Il numero di esempi processati contemporaneamente (batch) durante il training.
# Batch piccoli possono ridurre il consumo di memoria ma aggiornano i pesi con gradienti più rumorosi.

#pretrain_dataset="wikitext-103-raw-v1"
pretrain_dataset="wikitext-2-raw-v1"
instruct_dataset="yahma/alpaca-cleaned"
# -------------------------------
# Percorsi dei file necessari
# -------------------------------
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

encoder_path = "encoder.json"  
# Percorso al file JSON contenente il vocabolario: una mappatura dai token alle loro ID.

bpe_path = "vocab.bpe"  
# Percorso al file che contiene le regole di merge per il Byte-Pair Encoding (BPE).

model_path = "llm_instruct.pt"  
# Percorso in cui verrà salvato il modello fine-tuned (instruct) dopo il training.

pretrained_model_path = "pretrained_llm.pt"  
# Percorso in cui verrà salvato il modello pre-trained, prima del fine-tuning.

# -------------------------------
# Impostazione del device per l'addestramento/inferenza
# -------------------------------
# Su Mac con chip M3, invece di CUDA (non disponibile), PyTorch utilizza il backend MPS.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Stampa il device in uso (utile per verificare l'ambiente di esecuzione)
#print("Using device:", device)
import torch
from tokenizer import Tokenizer
from model import LLM
import config
# ------------------------------------------------------------------------------
# Funzione load_model()
# Questa funzione carica il tokenizer e il modello LLM, assegnandoli al device configurato.
# Inoltre, carica lo stato dei pesi salvati nel file specificato in config.model_path.
# ------------------------------------------------------------------------------
def load_model():
    # Inizializza il Tokenizer utilizzando i percorsi per encoder.json e vocab.bpe definiti in config.
    tokenizer = Tokenizer(config.encoder_path, config.bpe_path)
    
    # Determina la dimensione del vocabolario a partire dal dizionario del tokenizer.
    vocab_size = len(tokenizer.encoder)
    
    # Inizializza il modello LLM con i parametri specificati in config:
    # - vocab_size: dimensione del vocabolario
    # - config.d_model: dimensione degli embedding e dei layer interni
    # - config.nhead: numero di teste nel meccanismo di multi-head attention
    # - config.num_layers: numero di strati (blocchi Transformer)
    # - config.max_seq_len: lunghezza massima della sequenza
    # - config.dropout: tasso di dropout per la regularizzazione
    model = LLM(vocab_size, config.d_model, config.nhead, config.num_layers, config.max_seq_len, config.dropout)
    
    # Sposta il modello sul device configurato (ad esempio, "mps" per Mac M3 o "cpu" se MPS non è disponibile).
    model = model.to(config.device)

    # Carica il checkpoint dei pesi salvati per il modello fine-tuned.
    # torch.load utilizza il parametro map_location per assicurarsi che i pesi vengano caricati sul device corretto.
    checkpoint = torch.load(config.model_path, map_location=config.device)
    
    # Aggiorna lo stato del modello con i pesi caricati dal checkpoint.
    model.load_state_dict(checkpoint)
    
    # Imposta il modello in modalità "evaluation" (inference). Questo disabilita ad esempio il dropout,
    # garantendo che le predizioni siano deterministiche.
    model.eval()
    
    # Restituisce il tokenizer e il modello caricato.
    return tokenizer, model

# ------------------------------------------------------------------------------
# Se lo script viene eseguito direttamente, viene eseguita la parte di inferenza.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Carica il tokenizer e il modello utilizzando la funzione load_model().
    tokenizer, model = load_model()
    
    # Definisce un prompt di esempio per l'inferenza.
    # In questo caso, il prompt chiede al modello di scrivere una breve poesia sul tramonto.
    prompt = "Instruction: Write a poem about AI.\nResponse:\n"
    
    # Utilizza il tokenizer per convertire il prompt in una sequenza di ID (tokenizzazione).
    # Il risultato è una lista di interi, che viene incapsulata in un tensore.
    # Il tensore viene poi spostato sul device configurato (es. "mps" o "cpu").
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(config.device)
    
    # Genera una sequenza di output usando il metodo generate() del modello.
    # - input_ids: il prompt tokenizzato
    # - max_new_tokens=50: numero di token da generare in aggiunta al prompt
    # - temperature=1.0: parametro che controlla la casualità della generazione (1.0 indica sampling standard)
    generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=1.0)
    
    # Decodifica la sequenza generata (lista di token ID) in una stringa leggibile.
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    
    # Verifica che "Response:" sia presente nel testo e lo splitta.
    if "Response:" in generated_text:
        # Split del testo in corrispondenza della parola "Response:" e prendi la parte dopo.
        response_text = generated_text.split("Response:")[1].strip()
        print(response_text)
    else:
        print(generated_text)

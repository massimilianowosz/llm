import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------------
# Classe WikiTextDataset
# ---------------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        """
        Costruttore del dataset per il pre-training basato su WikiText-2 (o dataset simili).
        
        Args:
            texts (list): Lista di stringhe, ad esempio i testi presenti nel campo "text"
                          di un dataset come WikiText-2.
            tokenizer (Tokenizer): Istanza del tokenizer (ad es. della classe Tokenizer) che
                                   verrà utilizzato per convertire il testo in una sequenza
                                   di ID (tokenizzazione).
            max_length (int): Lunghezza massima della sequenza tokenizzata. Se una sequenza
                              supera questo limite, verrà troncata.
        """
        self.tokenizer = tokenizer        # Salviamo l'istanza del tokenizer per usarlo in seguito.
        self.max_length = max_length      # Salviamo il valore della lunghezza massima della sequenza.
        self.data = []                    # Inizializziamo una lista vuota per contenere le sequenze tokenizzate.
        
        # Cicliamo su ogni testo presente nella lista "texts"
        for text in texts:
            text = text.strip()           # Rimuoviamo eventuali spazi o caratteri di nuova linea in eccesso.
            if text == "":                # Se il testo è vuoto, saltiamo questo esempio.
                continue
            # Usiamo il tokenizer per convertire il testo in una lista di ID (token)
            token_ids = self.tokenizer.encode(text)
            # Se la sequenza tokenizzata è più lunga del massimo consentito, la troncamo.
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            # Aggiungiamo la sequenza (lista di token) alla lista dei dati.
            self.data.append(token_ids)

    def __len__(self):
        """
        Ritorna il numero di esempi nel dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Ritorna l'esempio (sequenza di token) all'indice idx come un tensore di tipo long.
        """
        return torch.tensor(self.data[idx], dtype=torch.long)

# ---------------------------------------------------------------------------------
# Classe InstructDataset
# ---------------------------------------------------------------------------------
class InstructDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        """
        Costruttore del dataset per l'instruct-tuning, ad esempio utilizzando il dataset Alpaca.
        
        Args:
            data (list): Lista di esempi in cui ogni esempio è un dizionario con le chiavi:
                         - "instruction": il testo dell'istruzione.
                         - "input": (opzionale) eventuale input aggiuntivo.
                         - "output": la risposta attesa.
            tokenizer (Tokenizer): Istanza del tokenizer usato per convertire il prompt in ID.
            max_length (int): Lunghezza massima della sequenza tokenizzata.
        """
        self.data = data                # Salviamo la lista di esempi.
        self.tokenizer = tokenizer      # Salviamo il tokenizer.
        self.max_length = max_length    # Salviamo la lunghezza massima.

    def __len__(self):
        """
        Ritorna il numero di esempi nel dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Ritorna il prompt tokenizzato per l'esempio all'indice idx.
        Il prompt viene costruito concatenando "instruction", "input" (se presente) e "output".
        """
        example = self.data[idx]   # Otteniamo l'esempio (dizionario) all'indice idx.
        # Otteniamo i campi; usiamo .get() per gestire eventuali assenze e rimuoviamo gli spazi in eccesso.
        instruction = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        output = example.get("output", "").strip()
        # Se esiste un input aggiuntivo, lo includiamo nel prompt; altrimenti usiamo solo instruction e output.
        if inp:
            prompt = f"Instruction: {instruction}\nInput: {inp}\nResponse:\n{output}"
        else:
            prompt = f"Instruction: {instruction}\nResponse:\n{output}"
        # Convertiamo il prompt in una sequenza di token (lista di ID) usando il tokenizer.
        token_ids = self.tokenizer.encode(prompt)
        # Se la lunghezza della sequenza supera il massimo, la troncamo.
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        # Restituiamo la sequenza come tensore di tipo long.
        return torch.tensor(token_ids, dtype=torch.long)

# ---------------------------------------------------------------------------------
# Funzione collate_fn
# ---------------------------------------------------------------------------------
def collate_fn(batch, pad_token_id):
    """
    Funzione di collate per il DataLoader. Serve ad unire un batch di sequenze di lunghezza variabile 
    in un singolo tensore, effettuando il padding delle sequenze più corte con il token di padding.
    
    Args:
        batch (list): Lista di tensori (sequenze) ottenuti da __getitem__ dei dataset.
        pad_token_id (int): ID del token utilizzato per il padding.
        
    Returns:
        Tensor: Un tensore di dimensione (batch_size, max_length_del_batch) contenente le sequenze
                con padding.
    """
    from torch.nn import functional as F  # Importiamo F per usare il padding
    lengths = [x.size(0) for x in batch]     # Calcoliamo la lunghezza di ogni sequenza nel batch.
    max_len = max(lengths)                   # Determiniamo la lunghezza massima nel batch.
    # Per ogni sequenza, se è più corta del max_len, la paddiamo con il token di padding (pad_token_id).
    padded = [F.pad(x, (0, max_len - x.size(0)), value=pad_token_id) for x in batch]
    # Uniamo le sequenze paddate in un unico tensore 2D (batch_size, max_len).
    return torch.stack(padded)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Classe PositionalEncoding
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Inizializza il modulo di codifica posizionale.

        Args:
            d_model (int): La dimensione degli embedding (vettore di rappresentazione per ogni token).
            dropout (float): Il tasso di dropout da applicare per regularizzare.
            max_len (int): La lunghezza massima delle sequenze per cui si pre-calcolano le posizioni.
        
        Teoria:
            I Transformer non hanno alcuna struttura ricorrente o convoluzionale per gestire
            la posizione dei token. Quindi si aggiungono codifiche posizionali agli embedding
            per dare al modello informazioni sulla posizione relativa dei token nella sequenza.
            Le codifiche posizionali usate qui sono funzioni sinusoidali che forniscono una
            rappresentazione continua delle posizioni.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crea una matrice "pe" di dimensioni (max_len, d_model) inizializzata a zero.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        
        # Crea un vettore "position" contenente gli indici di posizione da 0 a max_len-1,
        # e lo trasforma in un tensore colonna (max_len, 1).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calcola un termine di divisione per le frequenze, usato per generare le funzioni sinusoidali.
        # L'uso di exp(-log(10000)/d_model * (0, 2, 4, ...)) produce frequenze differenti per ogni dimensione.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        # Applica la funzione seno alle dimensioni pari e il coseno a quelle dispari.
        # In questo modo, ogni posizione ha una rappresentazione unica basata su funzioni sinusoidali.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Aggiunge una dimensione in testa per permettere la diffusione del batch:
        # ora pe ha forma (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Registra "pe" come buffer del modulo, in modo che non sia considerato un parametro addestrabile.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Aggiunge la codifica posizionale agli input e applica dropout.

        Args:
            x (Tensor): Input di forma (batch_size, seq_len, d_model)

        Returns:
            Tensor: L'input arricchito con le informazioni posizionali e dropout applicato.
        """
        # Aggiunge la codifica posizionale: si somma l'embedding posizionale alle rappresentazioni.
        x = x + self.pe[:, :x.size(1)]
        # Applica dropout per regularizzazione.
        return self.dropout(x)

# =============================================================================
# Classe TransformerBlock
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Inizializza un blocco Transformer composto da un modulo di self-attention e una rete feed-forward.

        Args:
            d_model (int): Dimensione degli embedding.
            nhead (int): Numero di teste nel meccanismo di multi-head attention.
            dropout (float): Tasso di dropout usato in vari punti del blocco.
        
        Teoria:
            Il blocco Transformer è la pietra angolare dei modelli come GPT e BERT. Esso
            combina self-attention (per catturare le relazioni tra token in una sequenza) e
            una rete feed-forward applicata individualmente a ciascun token, con connessioni
            residuali e normalizzazione per stabilizzare l'addestramento.
        """
        super(TransformerBlock, self).__init__()
        
        # Modulo di Multi-Head Self-Attention.
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Primo layer lineare della rete feed-forward che espande la dimensione a 4 * d_model.
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        
        # Un layer dropout per la rete feed-forward.
        self.dropout = nn.Dropout(dropout)
        
        # Secondo layer lineare che riduce la dimensione di nuovo a d_model.
        self.linear2 = nn.Linear(4 * d_model, d_model)
        
        # Normalizzazione Layer per stabilizzare il training dopo la self-attention.
        self.norm1 = nn.LayerNorm(d_model)
        # Normalizzazione Layer per stabilizzare il training dopo la rete feed-forward.
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        Esegue una forward pass del blocco Transformer.

        Args:
            src (Tensor): Input di forma (seq_len, batch_size, d_model)
            src_mask (Tensor, opzionale): Maschera causale per la self-attention.
        
        Returns:
            Tensor: Output della stessa forma dell'input, con informazioni contestuali arricchite.
        
        Teoria:
            - Self-Attention: Per ogni token, il modello calcola un vettore che rappresenta l'attenzione
              sui token della sequenza (usando query, key e value). Il meccanismo multi-head permette al
              modello di catturare diversi aspetti delle relazioni.
            - Residual Connection & LayerNorm: Viene aggiunto l'input originale all'output della self-attention
              e normalizzato, migliorando la stabilità del training.
            - Feed-Forward Network: Una rete neurale completamente connessa che applica una trasformazione non lineare
              ad ogni posizione, seguita da un'altra residual connection e normalizzazione.
        """
        # Applica la self-attention. 'attn_output' ha forma (seq_len, batch_size, d_model).
        attn_output, _ = self.attn(src, src, src, attn_mask=src_mask)
        
        # Applica una residual connection sommando l'input originale al risultato dell'attention,
        # seguita da una normalizzazione layer.
        src = self.norm1(src + attn_output)
        
        # Passa l'input normalizzato attraverso il primo layer lineare della rete feed-forward.
        # F.gelu è la funzione di attivazione (Gaussian Error Linear Unit) usata per introdurre non linearità.
        ff_intermediate = self.linear1(src)
        ff_intermediate = F.gelu(ff_intermediate)
        
        # Applica dropout per regularizzazione e passa l'output attraverso il secondo layer lineare.
        ff_output = self.linear2(self.dropout(ff_intermediate))
        
        # Aggiunge nuovamente una residual connection (input + output del feed-forward) e normalizza.
        src = self.norm2(src + ff_output)
        return src

# =============================================================================
# Classe LLM (Large Language Model)
# =============================================================================
class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len, dropout=0.1):
        """
        Inizializza il modello di linguaggio (LLM) basato su Transformer.

        Args:
            vocab_size (int): Numero di token nel vocabolario.
            d_model (int): Dimensione degli embedding.
            nhead (int): Numero di teste nella multi-head attention.
            num_layers (int): Numero di blocchi Transformer.
            max_seq_len (int): Lunghezza massima delle sequenze di input.
            dropout (float): Tasso di dropout.
        
        Teoria:
            Il modello segue l'architettura dei Transformer come quella di GPT. Le componenti principali
            sono:
              - Embedding dei token: Trasforma gli ID dei token in vettori continui.
              - Codifica posizionale: Aggiunge informazioni sulla posizione alla rappresentazione dei token.
              - Stack di blocchi Transformer: Ogni blocco applica self-attention e una rete feed-forward,
                consentendo al modello di apprendere relazioni complesse nel testo.
              - Proiezione finale: Un layer lineare che mappa le rappresentazioni finali nello spazio del vocabolario,
                producendo i logits per ciascun token.
        """
        super(LLM, self).__init__()
        
        # Embedding layer che trasforma gli ID dei token (interi) in vettori continui di dimensione d_model.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Modulo per aggiungere la codifica posizionale agli embedding.
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Stack di blocchi Transformer. Utilizziamo nn.ModuleList per gestire una lista di layer.
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        
        # Layer di normalizzazione finale: aiuta a stabilizzare le attivazioni prima della proiezione finale.
        self.ln_f = nn.LayerNorm(d_model)
        
        # Layer lineare finale ("head") che proietta lo spazio d_model nello spazio del vocabolario.
        # Bias viene disabilitato, in quanto spesso non è necessario per la proiezione finale.
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Salva la lunghezza massima della sequenza per riferimento.
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        """
        Esegue la forward pass del modello.

        Args:
            x (Tensor): Tensore di input di forma (batch_size, seq_len) contenente gli ID dei token.
            mask (Tensor, opzionale): Maschera causale per il self-attention, di forma (seq_len, seq_len).

        Returns:
            Tensor: Logits del modello, di forma (batch_size, seq_len, vocab_size).

        Procedura:
            1. Gli ID dei token vengono trasformati in embedding e scalati.
            2. Viene aggiunta la codifica posizionale.
            3. Le sequenze vengono trasposte per adattarsi alla dimensione richiesta dal modulo di Multi-head Attention.
            4. Viene applicato lo stack di blocchi Transformer.
            5. Viene applicata una normalizzazione layer finale.
            6. Le sequenze vengono trasposte nuovamente e passate attraverso il layer lineare per produrre i logits.
        """
        # Trasforma gli ID dei token in vettori di embedding e li scala con la radice quadrata della dimensione dell'embedding.
        # Lo scaling stabilizza i gradienti durante l'addestramento.
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        
        # Aggiunge la codifica posizionale agli embedding per fornire informazioni sul posizionamento.
        x = self.pos_encoder(x)
        
        # I moduli Multi-head Attention in PyTorch aspettano gli input con dimensione (seq_len, batch_size, d_model),
        # quindi trasponiamo il tensore da (batch_size, seq_len, d_model) a (seq_len, batch_size, d_model).
        x = x.transpose(0, 1)
        
        # Applica ogni blocco Transformer in sequenza. Ogni blocco aggiorna le rappresentazioni in base all'attenzione e alla rete feed-forward.
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        # Applica la normalizzazione finale sui vettori risultanti.
        x = self.ln_f(x)
        
        # Riporta il tensore alla forma originale (batch_size, seq_len, d_model).
        x = x.transpose(0, 1)
        
        # Applica il layer "head" che proietta le rappresentazioni in uno spazio di dimensione vocab_size,
        # ottenendo i logits per ogni token. Questi logits vengono usati per calcolare le probabilità sui token.
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """
        Metodo per la generazione autoregressiva di testo.

        Args:
            input_ids (Tensor): Tensore di forma (1, seq_len_iniziale) contenente gli ID dei token del prompt.
            max_new_tokens (int): Numero massimo di token da generare oltre il prompt.
            temperature (float): Fattore di scala per controllare la casualità del sampling.
                                   Valori inferiori a 1 rendono il modello più deterministico,
                                   valori superiori a 1 aumentano la varietà delle risposte.

        Procedura:
            1. Il modello viene posto in modalità evaluation (senza dropout, ecc.).
            2. Partendo dal prompt, il modello genera token uno alla volta in maniera autoregressiva.
            3. Ad ogni iterazione viene calcolata una maschera causale per evitare che il modello "guardi"
               token futuri.
            4. I logits per l'ultimo token vengono scalati in base alla temperatura e trasformati in probabilità
               tramite softmax.
            5. Viene effettuato il campionamento (multinomial) per scegliere il prossimo token.
            6. Il token generato viene concatenato alla sequenza e il processo si ripete fino a generare
               il numero desiderato di token.
        Returns:
            Tensor: Sequenza generata (comprensiva del prompt iniziale e dei nuovi token), di forma (1, seq_len_finale).
        """
        # Imposta il modello in modalità evaluation (disabilita dropout e altre modalità specifiche del training).
        self.eval()
        # Crea una copia del prompt per iniziare la generazione.
        generated = input_ids.clone()
        
        # Itera per il numero di token da generare
        for _ in range(max_new_tokens):
            # Determina la lunghezza attuale della sequenza generata.
            seq_len = generated.shape[1]
            # Crea una maschera causale (triangolare superiore) di forma (seq_len, seq_len) con -inf sopra la diagonale,
            # in modo da impedire al modello di utilizzare informazioni future durante la generazione.
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(generated.device)
            # Calcola i logits per la sequenza generata finora.
            logits = self(generated, mask)
            # Seleziona i logits dell'ultimo token della sequenza e scala in base alla temperatura.
            logits = logits[:, -1, :] / temperature
            # Applica la funzione softmax per ottenere una distribuzione di probabilità sui token del vocabolario.
            probs = F.softmax(logits, dim=-1)
            # Effettua il campionamento multinomiale per scegliere il prossimo token.
            next_token = torch.multinomial(probs, num_samples=1)
            # Concatenare il token appena generato alla sequenza.
            generated = torch.cat((generated, next_token), dim=1)
        return generated
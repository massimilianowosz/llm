import json
import re

def bytes_to_unicode():
    """
    Crea una mappatura da byte (0-255) a caratteri unicode "visibili".

    Spiegazione:
    - L'idea è trasformare ogni byte (numero intero da 0 a 255) in un carattere Unicode
      in modo che ogni byte abbia una rappresentazione "visibile". Questo è essenziale per
      operare in modalità "byte-level", come fa GPT-2, in cui si desidera preservare tutte le
      informazioni, anche per caratteri non ASCII.
    - Vengono selezionati i byte già "visibili" (ad esempio, i caratteri con codici da 33 a 126,
      e alcune altre gamme) e per i byte restanti si assegna un valore a partire da 256.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]  # Copia dei byte "visibili"
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # Converte i codici in caratteri Unicode
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Data una lista di simboli (ad es. caratteri o subword), restituisce l'insieme
    di coppie adiacenti.

    Spiegazione:
    - Per esempio, se word = ['h', 'e', 'l', 'l', 'o'], il risultato sarà:
      {('h','e'), ('e','l'), ('l','l'), ('l','o')}.
    - Queste coppie sono utilizzate nell'algoritmo BPE per decidere quali simboli unire.
    """
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i+1]))
    return pairs

class Tokenizer:
    def __init__(self, encoder_path, bpe_path):
        """
        Inizializza il tokenizer.

        Args:
            encoder_path (str): percorso al file encoder.json, contenente il vocabolario (token -> id).
            bpe_path (str): percorso al file vocab.bpe, contenente le regole di merge per il Byte-Pair Encoding (BPE).

        Spiegazione:
        - Il vocabolario viene caricato da un file JSON e viene creato anche il mapping inverso (decoder).
        - Le regole BPE vengono caricate dal file vocab.bpe e convertite in un dizionario che mappa ogni coppia
          (bigramma) al suo indice di priorità (bpe_ranks).
        - Viene inizializzata una cache per memorizzare i risultati già computati dall'algoritmo BPE,
          in modo da velocizzare operazioni ripetute.
        - Vengono create le mappature byte-level, utili per convertire il testo in una rappresentazione "visibile"
          che preserva ogni byte.
        - Infine, si definisce un pattern regex per suddividere il testo in token grezzi, simile a quello usato in GPT‑2.
        """
        # Carica il vocabolario dal file JSON
        with open(encoder_path, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        # Crea il decoder: mappa ID -> token
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Carica le regole BPE dal file vocab.bpe
        with open(bpe_path, 'r', encoding='utf-8') as f:
            bpe_data = f.read().split('\n')
        # La prima riga è un header; le righe successive contengono le regole di merge (coppie di token da unire)
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data[1:] if merge_str.strip()]
        # Crea un dizionario che assegna a ciascuna coppia un indice (priorità di merge)
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # Inizializza la cache per memorizzare i risultati dell'algoritmo BPE
        self.cache = {}
        
        # Crea le mappature byte-level: da byte a carattere "visibile"
        self.byte_encoder = bytes_to_unicode()
        # Crea la mappatura inversa: da carattere "visibile" al byte originale
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Definisce il pattern regex per la tokenizzazione (versione semplificata di quella usata in GPT‑2)
        self.pat = re.compile(
            r"""('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+)"""
        )

    def bpe(self, token):
        """
        Applica l'algoritmo BPE (Byte-Pair Encoding) a un singolo token.

        Args:
            token (str): Un token in forma di stringa, già trasformato a livello di byte.

        Returns:
            str: Il token trasformato dopo l'applicazione del BPE, dove alcuni simboli sono stati uniti.

        Spiegazione:
        - L'algoritmo BPE unisce iterativamente la coppia di simboli con la massima priorità di merge,
          secondo le regole definite in self.bpe_ranks.
        - Se il token è già stato elaborato in precedenza, il risultato viene restituito dalla cache.
        """
        # Controlla se il token è già stato elaborato e memorizzato nella cache.
        if token in self.cache:
            return self.cache[token]
        # Divide il token in una lista di caratteri.
        word = list(token)
        # Ottiene l'insieme delle coppie adiacenti.
        pairs = get_pairs(word)
        if not pairs:
            return token
        # Itera fino a che non è possibile effettuare ulteriori merge.
        while True:
            # Trova la coppia (bigramma) con il rank minimo (più alta priorità di merge).
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # Se il bigramma non è presente nelle regole, interrompe il ciclo.
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # Processa il token cercando di unire la coppia trovata.
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                # Se trova la coppia (first, second) contigua, la unisce in un singolo simbolo.
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            # Se dopo l'unione il token è composto da un solo elemento, esce dal ciclo.
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # Unisce i simboli ottenuti in una stringa, separandoli con spazi.
        word_str = " ".join(word)
        # Memorizza il risultato nella cache per future chiamate.
        self.cache[token] = word_str
        return word_str

    def encode(self, text):
        """
        Tokenizza il testo in modo simile al tokenizer di GPT‑2.

        Procedura:
          1. Applica la regex per dividere il testo in token "grezzi".
          2. Per ogni token, converte il token in bytes (UTF-8) e lo trasforma in una rappresentazione
             "visibile" usando self.byte_encoder.
          3. Applica l'algoritmo BPE per unire i token in subword.
          4. Converte i subword nel corrispondente ID usando il vocabolario; se un token non esiste,
             usa l'ID associato a "<unk>".

        Args:
            text (str): Il testo da tokenizzare.

        Returns:
            list: Lista di interi, ciascuno rappresentante l'ID di un token.
        """
        bpe_tokens = []
        # Applica il pattern regex per suddividere il testo in token grezzi.
        for token in re.findall(self.pat, text):
            # Converte il token in una sequenza di bytes (UTF-8).
            token_bytes = token.encode("utf-8")
            # Trasforma ogni byte nel corrispondente carattere "visibile" usando la mappatura byte_encoder.
            token_transformed = "".join(self.byte_encoder[b] for b in token_bytes)
            # Applica l'algoritmo BPE al token trasformato e divide il risultato in subtoken.
            bpe_result = self.bpe(token_transformed).split(" ")
            # Aggiunge i subtoken alla lista dei token finali.
            bpe_tokens.extend(bpe_result)
        # Ottiene l'ID del token <unk> (unknown), usato come fallback per token non presenti nel vocabolario.
        unk_token = "<unk>"
        unk_id = self.encoder.get(unk_token)
        # Mappa ogni token BPE ottenuto nel suo ID usando il vocabolario; se non presente, usa unk_id.
        return [self.encoder.get(token, unk_id) for token in bpe_tokens]

    def decode(self, token_ids):
        """
        Ricostruisce il testo a partire da una lista di token IDs.

        Procedura:
          1. Converte ogni ID nel corrispondente token usando il decoder.
          2. Unisce i token in una stringa.
          3. Converte la stringa a livello di byte in testo leggibile usando la mappatura byte_decoder.

        Args:
            token_ids (list): Lista di interi rappresentanti gli ID dei token.

        Returns:
            str: Il testo decodificato.
        """
        # Converte ogni ID nel token corrispondente; se non presente, restituisce una stringa vuota.
        tokens = [self.decoder.get(token_id, "") for token_id in token_ids]
        # Unisce tutti i token in una singola stringa.
        text = "".join(tokens)
        # Per ogni carattere nella stringa, ottiene il byte originale usando byte_decoder e crea un bytearray.
        byte_array = bytearray([self.byte_decoder.get(ch, ord(ch)) for ch in text])
        # Decodifica il bytearray in una stringa UTF-8, sostituendo eventuali errori.
        return byte_array.decode("utf-8", errors="replace")

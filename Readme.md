# Transformer Autoregressive Language Model

This project features an autoregressive Transformer-based language model developed for research and educational purposes. Drawing inspiration from state-of-the-art models, it serves as a hands-on tool for exploring advanced natural language processing techniques.

## Usage

### 1. Pre-training Phase
- **Purpose**: Learn general language representations from a large, unlabeled corpus
- **Process**: 
  - Model learns basic linguistic patterns and knowledge
  - Uses unsupervised learning on raw text data
  - Builds foundational understanding of language structure
- **Execute**: 
  To run the pre-training phase, execute:
  ```
  python pretrain.py
  ```

### 2. Instruction Fine-tuning Phase
- **Purpose**: Adapt the model to follow specific instructions and improve task performance
- **Process**:
  - Train on carefully curated instruction-based datasets
  - Teaches model to understand and execute precise user directives
  - Enhances model's ability to generate contextually appropriate responses
  - Improves zero-shot and few-shot learning capabilities
- **Execute**:
  To run the instruct-train phase, execute:
  ```
  python instruct_train.py
  ```

### 3. Inference Phase
- **Purpose**: Deploy the trained model to generate text or solve specific tasks
- **Process**:
  - Load pre-trained and fine-tuned model weights
  - Accept user prompts or input
  - Generate contextually relevant and instruction-aligned outputs
- **Execute**:
  To test the model, execute: 
  ```
  python inference.py
  ```

## Configuration
All model and training parameters are centralized in the `config.py` file. In this file, you can adjust settings to control various aspects of the model and its training process:

### Model Parameters:
- `d_model`: Dimension of token embeddings and internal representations
- `nhead`: Number of attention heads in the multi-head attention mechanism
- `num_layers`: Number of Transformer layers composing the model
- `max_seq_len`: Maximum length of input sequences (in tokens)
- `dropout`: Dropout rate for regularization

### Training Parameters:
- `learning_rate`: Learning rate for the optimizer
- `num_epochs_pretrain`: Number of epochs for unsupervised pre-training
- `num_epochs_instruct`: Number of epochs for instruction-based fine-tuning
- `batch_size`: Number of examples processed per training batch

### Dataset and File Paths:
Paths for datasets, vocabulary files (e.g., encoder.json, vocab.bpe), and directories where checkpoints and trained models are saved are also configurable.

Adjust these parameters in `config.py` to customize the model's behavior and training procedure according to your research or study needs.

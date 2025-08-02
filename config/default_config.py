import torch
config = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

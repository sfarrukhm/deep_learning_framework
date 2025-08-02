import torch
config = {
    'img_size': 128,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

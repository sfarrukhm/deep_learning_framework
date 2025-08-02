from models.simple_cnn import SimpleCNN
from datasets.mnist_loader import get_mnist_loaders
from training.trainer import train
from training.evaluator import evaluate
from utils.cnn_visualizer import plot_feature_maps, plot_filters
from utils.hooks import activation_outputs,get_activation
from config.default_config import config

import torch.nn as nn
import torch.optim as optim
import torch

def main():
    device=config['device']
    model = SimpleCNN().to(device)
    train_loader, test_loader = get_mnist_loaders(config['batch_size'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Register hooks for intermediate layer visualization
    model.conv1.register_forward_hook(get_activation['conv1'])
    model.conv2.register_forward_hook(get_activation['conv2'])

    # Train Model
    train(model, train_loader,criterion,optimizer,device,epochs=config['epochs'])
    plot_filters(model.conv1.weight, 'Conv1 Learned Filters')

    # run forward on one image to extract feature maps
    image, _ = next(iter(test_loader))
    image = image.to(device)
    model(image)

    plot_feature_maps(activation_outputs['conv1'], "Conv1 Feature Maps")
    plot_feature_maps(activation_outputs['conv2'], "Conv2 Feature Maps")



if __name__ == "__main__":
    main()

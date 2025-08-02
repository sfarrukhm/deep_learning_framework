import matplotlib.pyplot as plt

def plot_filters(tensor, title):
    num_kernels = tensor.shape[0]
    fig, axes = plt.subplots(1, num_kernels, figsize=(15, 5))
    for i in range(num_kernels):
        kernel = tensor[i, 0].cpu().detach()
        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'F{i}')
    fig.suptitle(title)
    plt.show()

def plot_feature_maps(features, title):
    num_maps = features.shape[1]
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    for i in range(num_maps):
        axes[i].imshow(features[0, i].detach().cpu(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'F{i}')
    fig.suptitle(title)
    plt.show()

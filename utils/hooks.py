activation_outputs = {}

def get_activation(name):
    def hook(model, input, output):
        activation_outputs[name] = output.detach()
    return hook

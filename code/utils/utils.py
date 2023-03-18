import numpy as np
def compute_sampling_threshold(global_step, k):
    ratio = k / (k + np.exp(global_step / k))
    return ratio

def print_model_parameters(model, only_num = True):
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
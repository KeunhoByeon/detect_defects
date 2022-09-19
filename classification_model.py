from torchvision.models.resnet import resnet18


def load_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet18(**kwargs)
    else:
        print('Model name {} is not implemented yet!'.format(model_name))
        raise TypeError

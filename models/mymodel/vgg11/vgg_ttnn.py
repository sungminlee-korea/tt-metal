import ttnn


class VGG11:
    def __init__(self, device, param, num_classes=1000):
        self.device = device
        self.conv1 = _conv2d()
        self.conv2 = _conv2d()
        self.conv3 = _conv2d()
        self.classifier = _classifier()

    def foward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


class _conv2d:
    def __init__(self):
        pass

    def __call__(self, x):
        pass
        return x


class _classifier:
    def __init__(self):
        pass

    def __call__(self, x):
        pass
        return x


def make_model():
    pass

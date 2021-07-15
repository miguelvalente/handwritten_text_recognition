from torchvision.transforms.functional import rotate

class Rotate(object):
    ''' Wrapper for torchvision.transforms.functional to use in Compose '''
    def __init__(self, angle=-90):
        self.angle = angle

    def __call__(self, img):
        img = rotate(img=img, angle=self.angle, expand=True)

        return img

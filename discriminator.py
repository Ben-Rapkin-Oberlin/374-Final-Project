from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, ReLU, Sigmoid
class Discriminator(Module):

    def __init__(self):
 
        super().__init__()
        w = 64
        
        
        self.dis = Sequential(
 
            # input is (3, 32, 32)
            Conv2d(in_channels = 3, out_channels = w, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32, 16, 16
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w, out_channels = w*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*2, 8, 8
            BatchNorm2d(w * 2),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*2, out_channels = w*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*4, 4, 4
            BatchNorm2d(w * 4),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*4, out_channels = w*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(w * 8),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*8, out_channels = w * 16, kernel_size = 4, stride = 2, padding = 1, bias=False),
            BatchNorm2d(w * 16),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = w*16, out_channels = w*32, kernel_size = 4, stride = 2, padding = 1, bias=False),
            BatchNorm2d(w * 32),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = w*32, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
            # ouput from above layer is b_size, 1, 1, 1
            Sigmoid()
        )
     
    def forward(self, input):
        return self.dis(input)
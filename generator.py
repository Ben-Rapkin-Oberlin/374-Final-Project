from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh
#import lazyload
class Generator(Module):
    def __init__(self):
 
        super().__init__()
        nz = 100
        ngf = 64

        self.gen = Sequential(
            
            Conv2d(in_channels = 6, out_channels =  ngf, kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf),
            LeakyReLU(inplace = True),

            Conv2d(in_channels = ngf, out_channels =  ngf * 2 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 2),
            LeakyReLU(inplace = True),

            Conv2d(in_channels = ngf * 2, out_channels =  ngf * 4 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 4),
            LeakyReLU(inplace = True),

            Conv2d(in_channels = ngf * 4, out_channels =  ngf * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 8),
            LeakyReLU(inplace = True),
            
            Conv2d(in_channels = ngf * 8, out_channels =  ngf * 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 16),
            LeakyReLU(inplace = True),

            Conv2d(in_channels = ngf * 16, out_channels =  1 , kernel_size = 4, stride = 1, padding = 0, bias = False),
            # #BatchNorm2d(num_features = ngf * 32),
            LeakyReLU(inplace = True),

            # ConvTranspose2d(in_channels = ngf*16, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
            # LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels = 1, out_channels =  ngf * 16 , kernel_size = 4, stride = 1, padding = 0, bias = False),    
            BatchNorm2d(num_features = ngf * 16), 
            ReLU(inplace = True),
            
            # ConvTranspose2d(in_channels = ngf *32, out_channels =  ngf * 16 , kernel_size = 4, stride = 2, padding = 1, bias = False),    
            # BatchNorm2d(num_features = ngf * 16), 
            # ReLU(inplace = True),

            ConvTranspose2d(in_channels = ngf * 16, out_channels =  ngf * 8 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 8), 
            ReLU(inplace = True),
            
            ConvTranspose2d(in_channels = ngf * 8, out_channels =  ngf * 4 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features =  ngf * 4),
            ReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 4, out_channels =  ngf * 2 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 2),
            ReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 2, out_channels =  ngf , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf),
            ReLU(inplace = True),

            ConvTranspose2d(in_channels = ngf, out_channels = 3 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            Tanh()
        )
 
    def forward(self, input):
        return self.gen(input)
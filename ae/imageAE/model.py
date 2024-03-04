from torch import nn

# N == Input dimension
# K == Kernel size
# S == Stride
# P == Padding
# OP == Output padding
# The formula for output dimension after a convolution is ((N-K+2P) / S) + 1
# The formula for output dimension after a max pooling is ((N-K) / S) + 1
# The formula for output dimension after a deconvolution is ((N-1)*S)) - (2*P) + K + OP
class ImageAutoencoder(nn.Module):

    def __init__(self, n_channels, height, width, latent_dim):
        super(ImageAutoencoder, self).__init__()
        
        self.conv_kernel = 4
        self.pool_kernel = 2
        self.stride = 2
        self.padding = 1

        #First Conv2D
        self.height_first_conv = ((height - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        self.width_first_conv =  ((width - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        print(f"After first conv: {self.height_first_conv} - {self.width_first_conv}")
        #First MaxPool2D
        self.height_first_pool = ((self.height_first_conv - self.pool_kernel) // self.stride) + 1 #21
        self.width_first_pool =  ((self.width_first_conv - self.pool_kernel) // self.stride) + 1 #21
        print(f"After first pool: {self.height_first_pool} - {self.width_first_pool}")
        #Second Conv2D
        self.height_second_conv = ((self.height_first_pool - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        self.width_second_conv =  ((self.width_first_pool - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        print(f"After second conv: {self.height_second_conv} - {self.width_second_conv}")
        #Second MaxPool2D
        self.height_second_pool = ((self.height_second_conv - self.pool_kernel) // self.stride) + 1 #5
        self.width_second_pool =  ((self.width_second_conv - self.pool_kernel) // self.stride) + 1 #5
        print(f"After second pool: {self.height_second_pool} - {self.width_second_pool}")
        #Third Conv2D
        #self.height_third_conv = ((self.height_second_pool - self.conv_kernel + (2*self.padding)) // 1) + 1 #4
        #self.width_third_conv =  ((self.width_second_pool - self.conv_kernel + (2*self.padding)) // 1) + 1 #4
        #print(f"After third conv: {self.height_third_conv} - {self.width_third_conv}")
        #Fourth Conv2D
        #self.height_fourth_conv = ((self.height_third_conv - self.conv_kernel + (2*self.padding)) // 1) + 1 #3
        #self.width_fourth_conv =  ((self.width_third_conv - self.conv_kernel + (2*self.padding)) // 1) + 1 #3
        #print(f"After fourth conv: {self.height_fourth_conv} - {self.width_fourth_conv}")
        #Third MaxPool2D
        #self.height_third_pool = ((self.height_fourth_conv - self.pool_kernel) // self.stride) + 1 #1
        #self.width_third_pool =  ((self.width_fourth_conv - self.pool_kernel) // self.stride) + 1 #1
        #print(f"After third max pool: {self.height_third_pool} - {self.width_third_pool}")
        #Input dimension for the linear layer
        #self.final_dim = latent_dim * self.height_third_pool * self.width_third_pool #128
        self.final_dim = latent_dim * self.height_second_pool * self.width_second_pool #3200
        print(f"Final dim: {self.final_dim}")

        # Encoder
        self.encoder = nn.Sequential(
            # First conv
            nn.Conv2d(n_channels, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # First pool
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            # Second conv
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            # Second pool
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            # Third conv
            #nn.Conv2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Fourth conv
            #nn.Conv2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Third pool
            #nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Flatten(),
            nn.Linear(self.final_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.final_dim),
            nn.ReLU(),
            nn.Unflatten(1, (latent_dim, self.height_second_pool, self.width_second_pool)), # Small CNN
            #nn.Unflatten(1, (latent_dim, self.height_third_pool, self.width_third_pool)), # Big CNN
            # Third pool
            #nn.Upsample(size=(self.height_fourth_conv, self.width_fourth_conv), mode='nearest'),
            # Fourth conv
            #nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Third conv
            #nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=self.conv_kernel, stride=1, padding=self.padding),
            #nn.ReLU(),
            # Second pool
            nn.Upsample(size=(self.height_second_conv, self.width_second_conv), mode='nearest'),
            # Second conv
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=self.padding),
            nn.ReLU(),
            # First pool
            nn.Upsample(size=(self.height_first_conv, self.width_first_conv), mode='nearest'),
            # First conv
            nn.ConvTranspose2d(latent_dim // 2, n_channels, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_encodings=False):
        x = self.encoder(x)
        #print(f"Encoder output size: {x.shape}")
        if not return_encodings:
            x = self.decoder(x)
            #print(f"Decoder output size: {x.shape}")
        return x


    

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
        self.height_final_dim = ((height - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        self.width_final_dim =  ((width - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #42
        #First MaxPool2D
        self.height_final_dim = ((self.height_final_dim - self.pool_kernel) // self.stride) + 1 #21
        self.width_final_dim =  ((self.width_final_dim - self.pool_kernel) // self.stride) + 1 #21
        #Second Conv2D
        self.height_final_dim = ((self.height_final_dim - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        self.width_final_dim =  ((self.width_final_dim - self.conv_kernel + (2*self.padding)) // self.stride) + 1 #10
        #Second MaxPool2D
        self.height_final_dim = ((self.height_final_dim - self.pool_kernel) // self.stride) + 1 #5
        self.width_final_dim =  ((self.width_final_dim - self.pool_kernel) // self.stride) + 1 #5
        #Input dimension for the linear layer
        self.final_dim = latent_dim * self.height_final_dim * self.width_final_dim #3200

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride),
            nn.Flatten(),
            nn.Linear(self.final_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.final_dim),
            nn.ReLU(),
            nn.Unflatten(1, (latent_dim, self.height_final_dim, self.width_final_dim)),
            nn.Upsample(scale_factor=self.pool_kernel, mode='nearest'),
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=self.padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.pool_kernel, mode='nearest'),
            nn.ConvTranspose2d(latent_dim // 2, n_channels, kernel_size=self.conv_kernel, stride=self.stride, padding=self.padding, output_padding=0),
            nn.ReLU()
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    

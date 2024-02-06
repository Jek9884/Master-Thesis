import torch
from torch import nn 

#TODO add random sampling ai frame dei video

class VideoAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VideoAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=input_shape[1], out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
        )
        
        gru_input_size = 16 * input_shape[2] * input_shape[3]
        # GRU layer to capture temporal information
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=latent_dim, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=latent_dim, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=input_shape[1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Identity(),
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        print("Encoder output shape:", x.shape)

        # Reshape for GRU
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        print("Reshaped for GRU shape:", x.shape)

        # GRU layer
        x, _ = self.gru(x.unsqueeze(1))  # Unsqueeze to add sequence dimension
        print("GRU output shape:", x.shape)

        # Reshape for decoder
        x = x.view(batch_size, self.gru.hidden_size, depth, height, width)
        print("Reshaped for Decoder shape:", x.shape)

        # Decoder
        x = self.decoder(x)
        print("Decoder output shape:", x.shape)

        return x


#def train(input_data, optimizer, loss, num_epochs, )
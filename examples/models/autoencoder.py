import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Bottleneck
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CombinedModel(nn.Module):
    def __init__(self, autoencoder, featurizer, classifier):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        decoded = self.autoencoder(x)  # Get encoded and decoded outputs
        features = self.featurizer(decoded)  # Featurize the decoded output
        output = self.classifier(features)  # Get final classification output
        return decoded, output  # Return all three outputs

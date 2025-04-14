class UNetSegmentation(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=3, lr=1e-3):
        super().__init__()
        self.lr = lr

        # Encoder (Downsampling)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Centre
        self.center = self._block(512, 1024)
        
        # Decoder (Upsampling)
        self.dec4 = self._block(1024+512, 512)
        self.dec3 = self._block(512+256, 256)
        self.dec2 = self._block(256+128, 128)
        self.dec1 = self._block(128+64, 64)
        
        # Couches finales
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling et upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Center
        center = self.center(self.pool(enc4))
        
        # Decoder avec skip connections
        dec4 = self.dec4(torch.cat([self.up(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2), enc1], dim=1))
        
        return self.final(dec1)

def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = nn.CrossEntropyLoss()(y_hat, y.squeeze(1).long())  # y.squeeze(1) si y est [B,1,H,W]
    
    self.log('train_loss', loss, prog_bar=True)
    return loss

def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = nn.CrossEntropyLoss()(y_hat, y.squeeze(1).long())
    
    # Calcul du Dice Score (métrique clé en segmentation)
    y_pred = torch.argmax(y_hat, dim=1)
    dice = self.dice_metric(y_pred, y)
    
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_dice', dice.mean(), prog_bar=True)  # Dice moyen sur le batch
    return loss

def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)
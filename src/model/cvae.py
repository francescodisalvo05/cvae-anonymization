import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import torch 

from torch import nn


class CEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super(CEncoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(input_dim + n_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),  # 2 for mean and variance.
        )
        
        self.norm = torch.distributions.Normal(0, 1)
        self.norm.loc = self.norm.loc.cuda()        # hack to get sampling on the GPU
        self.norm.scale = self.norm.scale.cuda()
        self.kl = 0
        

    def forward(self, x, y):
        x = self.encode(torch.cat((x,y), dim=-1))   # concatenate label to input (conditioning), then encode
        mean, log_var = torch.chunk(x, 2, dim=-1)
        sigma = torch.exp(log_var)
        z = mean + sigma * self.norm.sample(mean.shape)
        self.kl = (sigma**2 + mean**2 - log_var - 0.5).sum() 
        return z
    

class CDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        super(CDecoder, self).__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), 
            nn.Sigmoid()
        )
        

    def forward(self, x, y):
        return self.decode(torch.cat((x,y), dim=-1))


class CVAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes, device='cuda:0', learning_rate=0.001, beta=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = CEncoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = CDecoder(latent_dim, hidden_dim, input_dim, n_classes)


    def forward(self, x, y):
        z = self.encoder(x, y)
        return self.decoder(z, y)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_oh = F.one_hot(y, num_classes=self.hparams.n_classes).to(self.device)
        preds = self(x, y_oh)
        rec_loss = F.mse_loss(preds, x, reduction='sum')
        loss = rec_loss + self.hparams.beta * self.encoder.kl
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_oh = F.one_hot(y, num_classes=self.hparams.n_classes).to(self.device)
        preds = self(x, y_oh)
        rec_loss = F.mse_loss(preds, x, reduction='sum')
        loss = rec_loss + self.hparams.beta * self.encoder.kl
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    
    def generate_samples_v1(self, label_counts, train_min, train_max, var=1.0):
        """
        Generate a 1:1 replica in terms of size and label distribution.
        Thus, label_counts will be a dictionary containing {label:n_samples}

        This is used to fairly compare CVAE with kSAME, even though the
        difference might be minimal by sampling the labels according to the actual
        distribution (this is applied in generate_samples_v2).
        """
        gen_samples, gen_labels = [], []

        self.eval()  
        self.decoder.to(self.hparams.device) 

        with torch.no_grad():
            for label, n_samples in label_counts.items():
                latents = var * torch.randn((n_samples, self.hparams.latent_dim)).to(self.hparams.device) 
                label_tensor = torch.full((n_samples,), label, dtype=torch.long, device=self.hparams.device)
                y_oh = F.one_hot(label_tensor, num_classes=self.hparams.n_classes).to(self.hparams.device)
                x_hat = self.decoder(latents, y_oh).to(self.hparams.device)
                x_hat = self._un_normalize(x_hat, train_min, train_max)
                gen_samples.extend(x_hat.cpu().numpy())
                gen_labels.extend([label] * n_samples)
        return gen_samples, gen_labels
    
    def generate_samples_v2(self, label_distribution, num_samples, train_min, train_max, var=1.0):
        """
        Sample the labels according to the class categorical distribution.
        Thus, label_distribution is an array having in position "i", the probability of class "i".
        """
        gen_samples, gen_labels = [], []
        unique_labels = np.arange(len(label_distribution))

        self.eval()  
        self.decoder.to(self.hparams.device)
        
        with torch.no_grad():
            #for label, n_samples in label_distribution.items():
            for _ in range(num_samples):
                label = np.random.choice(unique_labels, p=label_distribution)
                latents = var * torch.randn((1, self.hparams.latent_dim)).to(self.hparams.device) 
                label_tensor = torch.full((1,), label, dtype=torch.long, device=self.hparams.device)
                y_oh = F.one_hot(label_tensor, num_classes=self.hparams.n_classes).to(self.hparams.device)
                x_hat = self.decoder(latents, y_oh).to(self.hparams.device)
                x_hat = self._un_normalize(x_hat, train_min, train_max)  # Assuming normalization method is adapted
                gen_samples.extend(x_hat.cpu().numpy())
                gen_labels.extend([label])
        return gen_samples, gen_labels


    def generate_one(self, label, train_min, train_max, var=1.0):
        self.eval()  # Ensure the model is in evaluation mode
        self.decoder.to(self.hparams.device) # put it on gpu

        with torch.no_grad():
            latents = var * torch.randn((1, self.hparams.latent_dim)).to(self.hparams.device) 
            label_tensor = torch.full((1,), label, dtype=torch.long, device=self.hparams.device)
            y_oh = F.one_hot(label_tensor, num_classes=self.hparams.n_classes).to(self.hparams.device)
            x_hat = self.decoder(latents, y_oh).to(self.hparams.device)
            x_hat = self._un_normalize(x_hat, train_min, train_max)  # Assuming normalization method is adapted

        return x_hat.cpu().numpy()[0]


    def _un_normalize(self, tensor, _min, _max):
        return (tensor) * (_max - _min) + _min
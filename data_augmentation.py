import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S IST')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Simplified WaveNet VAE with Increased Capacity
class WaveNetVAE(nn.Module):
    def __init__(self, latent_dim=512, timesteps=800, features=6):
        super(WaveNetVAE, self).__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.features = features

        # Encoder: Increased to 4 dilation layers
        self.conv_layers = nn.ModuleList()
        dilation_rates = [1, 2, 4, 8]  # Increased from 3 to 4 layers
        filters = 192  # Increased from 128 for better capacity
        for rate in dilation_rates:
            in_channels = features if rate == 1 else filters
            conv = nn.Conv1d(in_channels, filters, kernel_size=3, dilation=rate, padding=rate)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.LeakyReLU(0.1))
            self.conv_layers.append(nn.LayerNorm([filters, timesteps]))
            self.conv_layers.append(nn.Dropout(0.2))

        self.fc_mean = nn.Linear(filters * timesteps, latent_dim)
        self.fc_logvar = nn.Linear(filters * timesteps, latent_dim)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

        # Decoder: Increased capacity
        self.decoder_input = nn.Linear(latent_dim, filters * timesteps)
        nn.init.xavier_uniform_(self.decoder_input.weight)
        self.decoder_conv = nn.ModuleList()
        for rate in dilation_rates[::-1]:
            in_channels = filters if rate == 8 else filters
            conv = nn.ConvTranspose1d(in_channels, filters, kernel_size=3, dilation=rate, padding=rate, output_padding=0)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            self.decoder_conv.append(conv)
            self.decoder_conv.append(nn.LeakyReLU(0.1))
            self.decoder_conv.append(nn.LayerNorm([filters, timesteps]))
            self.decoder_conv.append(nn.Dropout(0.2))
        self.upconv = nn.ConvTranspose1d(filters, features, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.upconv.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, x):
        out = x.transpose(1, 2)  # [batch, timesteps, features] -> [batch, features, timesteps]
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv1d):
                out = layer(out)
            elif isinstance(layer, nn.LayerNorm):
                out = layer(out)
            elif isinstance(layer, (nn.LeakyReLU, nn.Dropout)):
                out = layer(out)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1)  # Flatten for FC
        return self.fc_mean(out), self.fc_logvar(out)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        out = self.decoder_input(z).view(-1, 192, 800)  # [batch, filters, timesteps]
        for layer in self.decoder_conv:
            if isinstance(layer, nn.ConvTranspose1d):
                out = layer(out)
            elif isinstance(layer, nn.LayerNorm):
                out = layer(out)
            elif isinstance(layer, (nn.LeakyReLU, nn.Dropout)):
                out = layer(out)
        out = self.upconv(out)
        return out.transpose(1, 2)  # [batch, timesteps, features]

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# FFT Loss
def compute_fft_loss(original, reconstructed, fs=400, freq_min=15, freq_max=160):
    freqs = fftfreq(original.shape[1], 1/fs)
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    fft_orig = torch.abs(torch.fft.fft(original, dim=1))
    fft_recon = torch.abs(torch.fft.fft(reconstructed, dim=1))
    fft_orig_filtered = fft_orig[:, mask]
    fft_recon_filtered = fft_recon[:, mask]
    return nn.functional.mse_loss(fft_orig_filtered, fft_recon_filtered)

# Training Function with Adjusted Parameters
def train_vae(vae, data, accel_max, gyro_max, epochs=1000, batch_size=4, plot_dir="C:/Users/Lee/Desktop/MultiSetup/Outputs/Training_Plots"):
    # Clear GPU memory
    torch.cuda.empty_cache()

    # Split data into train and validation (80-20)
    n_samples = data.shape[0]
    n_train = int(0.8 * n_samples)
    train_data = torch.FloatTensor(data[:n_train]).to(device)
    val_data = torch.FloatTensor(data[n_train:]).to(device)

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    os.makedirs(plot_dir, exist_ok=True)
    model_dir = os.path.dirname(plot_dir).replace("Training_Plots", "Models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"vae_{os.path.basename(plot_dir).replace('_', '')}.pth")

    best_val_loss = float('inf')
    patience, trials = 20, 0  # Increased patience to 20

    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch[0].to(device)
            recon, mean, logvar = vae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            fft_loss = compute_fft_loss(batch, recon)
            loss = recon_loss + 0.05 * kl_loss + 1.0 * fft_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                recon, mean, logvar = vae(batch)
                recon_loss = nn.functional.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                fft_loss = compute_fft_loss(batch, recon)
                loss = recon_loss + 0.05 * kl_loss + 1.0 * fft_loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample = train_data[0:1].to(device)
                recon_sample, _, _ = vae(sample)
                original = sample.cpu().numpy() * np.array([accel_max] * 3 + [gyro_max] * 3)
                reconstructed = recon_sample.cpu().numpy() * np.array([accel_max] * 3 + [gyro_max] * 3)
                plot_reconstruction(original, reconstructed, epoch + 1, plot_dir)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trials = 0
            try:
                torch.save(vae.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path} with val loss {best_val_loss:.4f}")
            except RuntimeError as e:
                logger.error(f"Failed to save model to {model_path}: {str(e)}")
        else:
            trials += 1
            if trials >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return vae

# Plotting Functions
def plot_reconstruction(original, reconstructed, epoch, plot_dir):
    axes = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    for i in range(6):
        axs[i].plot(original[0, :, i], label="Original", alpha=0.8)
        axs[i].plot(reconstructed[0, :, i], label="Reconstructed", alpha=0.6)
        axs[i].set_title(axes[i])
        axs[i].legend()
    fig.suptitle(f"Reconstruction at Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"reconstruction_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved reconstruction plot: {plot_path}")

def main():
    config = {
        "input_dir": "C:/Users/Lee/Desktop/MultiSetup/Dataset",
        "numeric_axes": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    }
    input_dir = config["input_dir"]
    axes = config["numeric_axes"]

    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return

    for condition in os.listdir(input_dir):
        condition_dir = os.path.join(input_dir, condition)
        if not os.path.isdir(condition_dir) or condition.lower() in ["augmented_data", "augmented"]:
            continue

        logger.info(f"Processing data for condition: {condition}")
        sample_files = [f for f in os.listdir(condition_dir) if f.endswith(".csv") and "aug" not in f]
        data_list = []
        for sample_file in sample_files:
            file_path = os.path.join(condition_dir, sample_file)
            data = pd.read_csv(file_path)
            if data.shape[0] != 800:
                logger.warning(f"Skipping {sample_file}: Incorrect timesteps {data.shape[0]}")
                continue
            if not all(ax in data.columns for ax in axes):
                logger.warning(f"Skipping {sample_file}: Missing columns")
                continue
            data_values = data[axes].values
            if np.any(np.isnan(data_values)) or np.any(np.isinf(data_values)):
                logger.warning(f"Skipping {sample_file}: Contains NaN or inf")
                continue
            # Apply condition-specific preprocessing
            if "Idle" in condition:
                scale = np.random.uniform(0.98, 1.02, size=(1, data_values.shape[1]))
                shift = np.random.randint(-2, 3)
            elif "Normal" in condition:
                scale = np.random.uniform(0.95, 1.05, size=(1, data_values.shape[1]))
                shift = np.random.randint(-5, 6)
            else:
                scale = np.random.uniform(0.9, 1.1, size=(1, data_values.shape[1]))
                shift = np.random.randint(-10, 11)
            data_values = np.roll(data_values * scale, shift, axis=0)
            data_values[:max(0, shift), :] = 0
            data_values[min(800, 800 + shift):, :] = 0
            data_list.append(data_values)

        if not data_list:
            logger.warning(f"No valid data for {condition}")
            continue

        data_array = np.stack(data_list, axis=0)
        accel_max = np.percentile(np.abs(data_array[:, :, :3]), 95)
        gyro_max = np.percentile(np.abs(data_array[:, :, 3:]), 95)
        data_tensor = data_array / np.array([accel_max] * 3 + [gyro_max] * 3)

        vae = WaveNetVAE().to(device)
        vae = train_vae(vae, data_tensor, accel_max, gyro_max,
                        plot_dir=f"C:/Users/Lee/Desktop/MultiSetup/Outputs/Training_Plots/{condition.replace(' ', '_')}")

if __name__ == "__main__":
    main()
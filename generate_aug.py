import torch
import torch.nn as nn
import numpy as np
import os
import logging
import pandas as pd
from scipy.fft import fft, fftfreq

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S IST')
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Enhanced WaveNetVAE class matching the trained model
class WaveNetVAE(nn.Module):
    def __init__(self, latent_dim=512, timesteps=800, features=6):
        super(WaveNetVAE, self).__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.features = features

        # Encoder: Deeper WaveNet with 9 dilation rates
        self.conv_layers = nn.ModuleList()
        dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        filters = 256
        for rate in dilation_rates:
            in_channels = features if rate == 1 else filters
            conv = nn.Conv1d(in_channels, filters, kernel_size=3, dilation=rate, padding=rate)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.LeakyReLU(0.1))
            self.conv_layers.append(nn.LayerNorm([filters, timesteps]))
            self.conv_layers.append(nn.Dropout(0.2))
            if rate > 1:
                self.conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size=1, padding=0))
                self.conv_layers.append(nn.LeakyReLU(0.1))

        # Multi-layer LSTM with Attention
        self.lstms = nn.LSTM(input_size=filters, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        nn.init.orthogonal_(self.lstms.weight_ih_l0)
        nn.init.orthogonal_(self.lstms.weight_hh_l0)
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4)  # 512 * 2 from bidirectional
        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder: Deeper WaveNet with transposed convolution
        self.decoder_input = nn.Linear(latent_dim, filters * timesteps)
        nn.init.xavier_uniform_(self.decoder_input.weight)
        self.decoder_conv = nn.ModuleList()
        for rate in dilation_rates[::-1]:
            in_channels = filters if rate == 256 else filters
            conv = nn.Conv1d(in_channels, filters, kernel_size=3, dilation=rate, padding=rate)
            self.decoder_conv.append(conv)
            self.decoder_conv.append(nn.LeakyReLU(0.1))
            self.decoder_conv.append(nn.LayerNorm([filters, timesteps]))
            self.decoder_conv.append(nn.Dropout(0.2))
            if rate < 256:
                self.decoder_conv.append(nn.Conv1d(in_channels, filters, kernel_size=1, padding=0))
                self.decoder_conv.append(nn.LeakyReLU(0.1))
        self.upconv = nn.ConvTranspose1d(filters, features, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.upconv.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, x):
        out = x.transpose(1, 2)  # [batch, timesteps, features] -> [batch, features, timesteps]
        residual = x.transpose(1, 2)
        for i, layer in enumerate(self.conv_layers):
            if isinstance(layer, nn.Conv1d):
                if i % 5 == 0 and i > 0:  # After conv, relu, norm, dropout
                    out = out + residual if out.shape == residual.shape else out  # Residual connection
                    residual = out
                out = layer(out)
            elif isinstance(layer, (nn.LayerNorm, nn.Dropout)):
                out = layer(out)
        out = out.transpose(1, 2)  # [batch, timesteps, filters] for LSTM
        out, _ = self.lstms(out)
        out = out.transpose(0, 1)  # [timesteps, batch, 1024] for attention
        attn_output, _ = self.attention(out, out, out)
        out = attn_output.transpose(0, 1)[:, -1, :]  # Take last timestep after attention
        return self.fc_mean(out), self.fc_logvar(out)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        out = self.decoder_input(z).view(-1, 256, 800)  # [batch, filters, timesteps]
        residual = out
        for i, layer in enumerate(self.decoder_conv):
            if isinstance(layer, nn.Conv1d):
                if i % 5 == 0 and i > 0:  # After conv, relu, norm, dropout
                    out = out + residual if out.shape == residual.shape else out  # Residual connection
                    residual = out
                out = layer(out)
            elif isinstance(layer, (nn.LayerNorm, nn.Dropout)):
                out = layer(out)
        return self.upconv(out).transpose(1, 2)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Adjusted data validation function with relaxed range and logging
def validate_sample(data, axes, expected_timesteps=800):
    """Strict validation of a single sample with relaxed range logging."""
    if data.shape[0] != expected_timesteps:
        raise ValueError(f"Sample length {data.shape[0]} does not match expected {expected_timesteps}")
    if not all(ax in data.columns for ax in axes):
        raise ValueError(f"Missing columns, expected {axes}, got {data.columns.tolist()}")
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Sample contains NaN or infinite values")
    if np.any(np.abs(data) > 100):  # Relaxed to 100 with warning
        logger.warning(f"Sample contains values > 100, consider normalization: max value {np.max(np.abs(data))}")
    return data

# Function to validate temporal patterns via FFT
def validate_temporal_patterns(original, augmented, fs=400, tolerance=0.1):
    """Check if augmented data preserves temporal patterns within tolerance."""
    fft_orig = np.abs(fft(original, axis=0))
    fft_aug = np.abs(fft(augmented, axis=0))
    freqs = fftfreq(original.shape[0], 1/fs)
    dominant_freq_orig = freqs[np.argmax(fft_orig, axis=0)]
    dominant_freq_aug = freqs[np.argmax(fft_aug, axis=0)]
    deviation = np.abs(dominant_freq_orig - dominant_freq_aug) / (dominant_freq_orig + 1e-8)
    if np.any(deviation > tolerance):
        logger.warning(f"Temporal pattern deviation exceeds {tolerance*100}%: max deviation {np.max(deviation*100):.2f}%")
        return False
    return True

# Data generation function with CSV output per sample and augmentations
def generate_augmented_data(model_path, input_dir, output_dir, condition, accel_max, gyro_max, batch_size=16, num_augmentations=50):
    # Load the pre-trained model
    vae = WaveNetVAE().to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    logger.info(f"Loaded model from {model_path}")

    # Strict input directory and file validation
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    sample_files = [f for f in os.listdir(input_dir) if f.endswith(".csv") and "aug" not in f]
    if not sample_files:
        raise ValueError(f"No valid input files found in {input_dir}")

    # Randomly select up to 50 samples
    num_samples = min(50, len(sample_files))
    selected_samples = np.random.choice(sample_files, num_samples, replace=False)

    # Load and validate data
    data_list = []
    axes = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    for sample_file in selected_samples:
        file_path = os.path.join(input_dir, sample_file)
        try:
            data = pd.read_csv(file_path)
            data = validate_sample(data, ["timestamp"] + axes)
            data_values = data[axes].values
            data_list.append(data_values)
        except Exception as e:
            logger.warning(f"Skipping {sample_file}: {str(e)}")
            continue

    if not data_list:
        raise ValueError(f"No valid data loaded from {input_dir}")

    data_array = np.stack(data_list, axis=0)
    data_tensor = torch.FloatTensor(data_array / np.array([accel_max] * 3 + [gyro_max] * 3)).to(device)

    # Define augmentation functions with small magnitudes (commented out)
    # def apply_scaling(data):
    #     scale_factor = np.random.uniform(0.9, 1.1, size=(data.shape[1],))
    #     return data * scale_factor
    #
    # def apply_rotation(data):
    #     theta = np.random.uniform(-5, 5) * np.pi / 180  # Small rotation in radians
    #     rot_matrix = np.array([
    #         [np.cos(theta), -np.sin(theta), 0],
    #         [np.sin(theta), np.cos(theta), 0],
    #         [0, 0, 1]
    #     ])
    #     rotated = np.zeros((data.shape[0], 3))  # [timesteps, 3] for accel_x, accel_y, accel_z
    #     for t in range(data.shape[0]):
    #         rotated[t] = np.dot(rot_matrix, data[t, :3])
    #     return np.hstack((rotated, data[:, 3:]))  # Keep gyro data unchanged
    #
    # def apply_time_shift(data):
    #     shift = np.random.randint(-799, 800)  # Limit shift to valid range (-799 to 799)
    #     shifted = np.zeros_like(data)
    #     if shift > 0:
    #         effective_shift = min(shift, data.shape[0] - 1)  # Cap shift to avoid empty slice
    #         if effective_shift > 0:
    #             shifted[effective_shift:, :] = data[:-effective_shift, :]
    #             shifted[:effective_shift, :] = 0  # Pad with zeros
    #         else:
    #             shifted[:, :] = data[:, :]  # No shift if capped to 0
    #     elif shift < 0:
    #         effective_shift = max(shift, -data.shape[0] + 1)  # Cap negative shift
    #         if effective_shift < 0:
    #             shifted[:effective_shift, :] = data[-effective_shift:, :]
    #             shifted[effective_shift:, :] = 0  # Pad with zeros
    #         else:
    #             shifted[:, :] = data[:, :]  # No shift if capped to 0
    #     else:
    #         shifted[:, :] = data[:, :]  # No shift if shift is 0
    #     return shifted

    # Generate and save augmented data per sample
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size].to(device)
            original_timestamps = pd.read_csv(os.path.join(input_dir, selected_samples[i // batch_size]))["timestamp"].values
            original_data = batch.cpu().numpy() * np.array([accel_max] * 3 + [gyro_max] * 3)
            for j in range(num_augmentations):
                recon, _, _ = vae(batch)
                if recon.shape != batch.shape:
                    raise ValueError(f"Reconstruction shape {recon.shape} does not match input shape {batch.shape}")
                recon_data = recon.cpu().numpy() * np.array([accel_max] * 3 + [gyro_max] * 3)

                # Apply augmentations (commented out)
                # for k in range(aug_data.shape[0]):  # Iterate over batch
                #     if j % 3 == 0:  # Scaling
                #         aug_data[k] = apply_scaling(aug_data[k])
                #     elif j % 3 == 1:  # Rotation
                #         aug_data[k] = apply_rotation(aug_data[k])
                #     else:  # Time Shift
                #         aug_data[k] = apply_time_shift(aug_data[k])

                # Validate temporal patterns for each sample in batch
                aug_data = recon_data  # Use VAE reconstruction directly
                for k in range(aug_data.shape[0]):
                    if not validate_temporal_patterns(original_data[k], aug_data[k], tolerance=0.1):
                        logger.warning(f"Augmented sample {j+1} for {selected_samples[i // batch_size]} (batch index {k}) discarded due to temporal pattern deviation")
                        aug_data[k] = original_data[k]  # Fallback to original if validation fails

                # Create DataFrame with timestamp and sensor data
                for k in range(aug_data.shape[0]):
                    aug_df = pd.DataFrame({
                        "timestamp": original_timestamps,
                        "accel_x": aug_data[k, :, 0],
                        "accel_y": aug_data[k, :, 1],
                        "accel_z": aug_data[k, :, 2],
                        "gyro_x": aug_data[k, :, 3],
                        "gyro_y": aug_data[k, :, 4],
                        "gyro_z": aug_data[k, :, 5]
                    })
                    output_file = os.path.join(output_dir, f"augmented_{os.path.splitext(selected_samples[i // batch_size])[0]}_aug{j+1}_batch{k+1}.csv")
                    aug_df.to_csv(output_file, index=False, float_format="%.6f")  # 6 decimals for all numeric columns
                    logger.info(f"Saved augmented data to {output_file}")
def main():
    config = {
        "input_dir_base": "C:/Users/Lee/Desktop/MultiSetup/Dataset",
        "model_dir": "C:/Users/Lee/Desktop/MultiSetup/Outputs/Models",
        "output_dir_base": "C:/Users/Lee/Desktop/MultiSetup/Dataset/augmented"
    }
    axes = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

    if not os.path.exists(config["model_dir"]):
        logger.error(f"Model directory {config['model_dir']} does not exist")
        return

    for condition in os.listdir(config["input_dir_base"]):
        condition_dir = os.path.join(config["input_dir_base"], condition)
        if not os.path.isdir(condition_dir) or condition.lower() in ["augmented_data", "augmented"]:
            continue

        logger.info(f"Generating data for condition: {condition}")
        model_file = os.path.join(config["model_dir"], f"vae_{''.join(word.capitalize() for word in condition.split())}.pth")
        if not os.path.exists(model_file):
            logger.warning(f"Model file {model_file} not found, skipping {condition}")
            continue

        sample_files = [f for f in os.listdir(condition_dir) if f.endswith(".csv") and "aug" not in f]
        if not sample_files:
            logger.warning(f"No input files found for {condition}, skipping")
            continue

        # Compute scaling factors from a sample (consistent with training)
        sample_data = pd.read_csv(os.path.join(condition_dir, sample_files[0]))
        sample_data = validate_sample(sample_data, ["timestamp"] + axes)
        accel_max = np.percentile(np.abs(sample_data[axes[:3]]), 95)
        gyro_max = np.percentile(np.abs(sample_data[axes[3:]]), 95)

        output_dir = os.path.join(config["output_dir_base"], condition.replace(' ', '_'))
        generate_augmented_data(model_path=model_file, input_dir=condition_dir, output_dir=output_dir, condition=condition, accel_max=accel_max, gyro_max=gyro_max, num_augmentations=50)

if __name__ == "__main__":
    main()
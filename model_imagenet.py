import torch
import torch.nn as nn
import torch.fft as fft

# 通道注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x size: (batch_size, channels, time_steps)
        b, c, t = x.size()
        avg_out = self.avg_pool(x).view(b, c)  # 输出维度 (batch_size, channels)
        avg_out = self.fc(avg_out).view(b, c, 1)  # 将它转换为 (batch_size, channels, 1)
        return x * avg_out.expand_as(x)  # 注意力乘以输入，调整形状后进行广播

# 频段注意力机制模块
class BandAttention(nn.Module):
    def __init__(self, num_bands):
        super(BandAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_bands, num_bands, bias=False),  # 保持输入和输出的频段数量一致
            nn.ReLU(inplace=True),
            nn.Linear(num_bands, num_bands, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x size: (batch_size, channels, num_bands)
        b, c, f = x.size()  # b: batch_size, c: channels, f: num_bands
        avg_out = x.mean(dim=1)  # 在通道维度上取平均，结果为 (batch_size, num_bands)
        attn = self.fc(avg_out)  # 注意力机制，输出维度为 (batch_size, num_bands)
        return x * attn.unsqueeze(1).expand_as(x)  # 广播以适应原始输入的形状



# EEG编码器模型
class EEGEncoder(nn.Module):
    def __init__(self, input_size=(128,512), hidden_size=128, num_layers=2, num_bands=4):
        super(EEGEncoder, self).__init__()
        
        # 时域特征提取（双向LSTM）
        self.lstm = nn.LSTM(input_size[0], hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 通道注意力机制
        self.channel_attention = ChannelAttention(input_size[1])
        
        # CNN用于频域特征提取
        self.conv1d = nn.Conv1d(in_channels=num_bands, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # 频段注意力机制
        self.band_attention = BandAttention(64)
        
        # 全连接层整合特征
        self.fc = nn.Linear(320, 40)
    
    def fft_band_extraction(self, x, sampling_rate):
        # 对输入的EEG信号进行快速傅里叶变换（FFT）
        fft_x = fft.rfft(x, dim=-1)
        freqs = fft.rfftfreq(x.size(-1), d=1.0 / sampling_rate)

        # 提取特定频段的频域特征（Theta, Alpha, Beta, Gamma）
        theta = (freqs >= 4) & (freqs <= 7)
        alpha = (freqs >= 8) & (freqs <= 13)
        beta = (freqs >= 14) & (freqs <= 29)
        gamma = (freqs >= 30) & (freqs <= 47)

        band_features = torch.cat([
            fft_x[:, :, theta].abs().mean(dim=-1, keepdim=True),
            fft_x[:, :, alpha].abs().mean(dim=-1, keepdim=True),
            fft_x[:, :, beta].abs().mean(dim=-1, keepdim=True),
            fft_x[:, :, gamma].abs().mean(dim=-1, keepdim=True)
        ], dim=-1)

        return band_features  # 返回形状为 (batch_size, channels, num_bands)
    
    def forward(self, x, sampling_rate=128):
        bs, electrodes, samples = x.size()

        # 通道注意力机制
        x_channel_attention = self.channel_attention(x.transpose(1, 2))  # 调整维度以适应通道注意力
        # 时域特征提取
        x_lstm, _ = self.lstm(x_channel_attention) # 输出维度为 (batch_size, samples, hidden_size * 2)
        
        x_lstm = x_lstm.mean(dim=1)  # 在时间维度上取平均
        
        # 频域特征提取
        x_fft = self.fft_band_extraction(x, sampling_rate)
        x_fft = self.conv1d(x_fft.transpose(1, 2))  # Conv1D expects (batch, channels, freq)
        x_fft = self.pool(x_fft)
        x_fft = self.band_attention(x_fft.transpose(1, 2))
        x_fft = x_fft.mean(dim=-1)  # 在频域维度上取平均
        
        # 拼接时域和频域特征
        x_combined = torch.cat([x_lstm, x_fft], dim=-1)
        
        # 最终特征输出
        output = self.fc(x_combined)
        
        return output


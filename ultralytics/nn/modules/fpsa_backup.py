import torch
import torch.nn as nn

__all__ = ('FPSA')


class FPSA(nn.Module):
    def __init__(self, in_channels, out_channels=None, *args, **kwargs):
        super(FPSA, self).__init__()
        out_channels = out_channels or in_channels

        # Frequency domain attention layers
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Phase shift attention layers
        self.phase_attn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.combine = nn.Conv2d(out_channels, out_channels, kernel_size=1)


    def forward(self, x):
        # Ensure input is at least float32 for FFT operations
        
        original_dtype = x.dtype
        if original_dtype == torch.float16:
            x = x.to(torch.float32)  # Convert to float32 for FFT operations

        # Disable AMP during FFT operations
        with torch.cuda.amp.autocast(enabled=False):
            # Apply FFT to get amplitude and phase
            fft = torch.fft.fft2(x, dim=(-2, -1))
            amp, phase = torch.abs(fft), torch.angle(fft)

            # Frequency and phase attention
            amp_attn = self.freq_attn(amp)
            phase_shift = self.phase_attn(phase)

            # Combine frequency and phase
            fft_combined = amp_attn * torch.exp(1j * phase_shift)
            spatial_output = torch.fft.ifft2(fft_combined, dim=(-2, -1)).real

        # Convert back to original data type if necessary
        if original_dtype == torch.float16:
            spatial_output = spatial_output.to(torch.float16)


        # Combine with original input 
        output = self.combine(spatial_output + x.to(original_dtype))

        output = self.combine(spatial_output + x)

        return output

import torch
import torch.nn as nn

__all__ = ('FPSA',)


class FPSA(nn.Module):
    def __init__(self, in_channels, out_channels=None, *args, **kwargs):
        super().__init__()
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

        # Final combine conv
        self.combine = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Unified forward for both training mode (float32) and eval mode (optionally half).
        1. FFT & IFFT in float32 (avoiding ComplexHalf issues)
        2. If self.training == False (eval mode), cast amp & phase to half before freq/phase attn
        3. Combine conv in matching dtype to avoid mismatch errors
        """

        original_dtype = x.dtype  # float16 or float32, etc.

        # --------------------------
        # (1) FFT in float32
        # --------------------------
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()  # if x is half, upcast to float32
            fft_out = torch.fft.fft2(x_fp32, dim=(-2, -1))
            amp_fp32, phase_fp32 = torch.abs(fft_out), torch.angle(fft_out)
            # amp_fp32, phase_fp32 are float32

        # --------------------------
        # (2) freq_attn, phase_attn
        #     - Train: use float32
        #     - Eval : use half
        # --------------------------
        if self.training:
            amp, phase = amp_fp32, phase_fp32
        else:
            amp = amp_fp32.half()
            phase = phase_fp32.half()

        # Run freq_attn & phase_attn in whichever dtype (float32 or half)
        amp_attn = self.freq_attn(amp)
        phase_shift = self.phase_attn(phase)

        # --------------------------
        # (3) IFFT in float32 again
        # --------------------------
        with torch.cuda.amp.autocast(enabled=False):
            # Cast back up to float32 for complex exp and IFFT
            fft_combined = amp_attn.float() * torch.exp(1j * phase_shift.float())
            spatial_output_fp32 = torch.fft.ifft2(fft_combined, dim=(-2, -1)).real  # real part only

        # If eval mode, we can cast the spatial output to half
        if not self.training:
            spatial_output = spatial_output_fp32.half()
        else:
            spatial_output = spatial_output_fp32

        # --------------------------
        # (4) skip connection + combine conv
        #     Must match dtype of combine's weight
        # --------------------------
        # If combine is half, we feed half. If combine is float32, we feed float32.
        # Typically, if training we do float32; if eval we do half. 
        # But you could unify it further or strictly enforce one dtype for the combine layer.

        # ensure x matches spatial_output dtype
        x_cast = x.to(spatial_output.dtype)
        out = self.combine(spatial_output + x_cast)

        return out

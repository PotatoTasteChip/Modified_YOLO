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

        # Final combine conv
        self.combine = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: float16(AMP) 또는 float32일 수 있음
        - FFT와 복소 연산은 float32에서 수행
        - Conv, BN 등은 모델 파라미터 dtype과 동일하게 수행
        """
        original_dtype = x.dtype  # 예: torch.float16 또는 torch.float32

        # (1) FFT 부분은 float32로 고정(복소 half 미완성 문제 방지)

        with torch.cuda.amp.autocast(enabled=False):
            fft_input = x.float()  # half → float32 변환
            fft_out = torch.fft.fft2(fft_input, dim=(-2, -1))
            amp, phase = torch.abs(fft_out), torch.angle(fft_out)
        # amp, phase는 여기서 float32 상태

        amp = torch.log1p(amp)  # 주파수 도메인 에서 amp 너무 크면(ex nan) -> log(amp+1)로 축소한다고 함 

        # (2) freq_attn, phase_attn를 수행하기 위해 amp, phase를 다시 original_dtype으로 캐스팅

        amp = amp.to(original_dtype)
        phase = phase.to(original_dtype)

        # freq_attn, phase_attn는 Conv + BN 등 파라미터가 original_dtype임
        amp_attn = self.freq_attn(amp)         # half 또는 float32
        phase_shift = self.phase_attn(phase)   # half 또는 float32


        # (3) 복소수 exponential + IFFT는 half 복소연산이 불안정하므로, 다시 float32로 올려서 계산 후, 최종적으로 original_dtype으로 내려줌
        with torch.cuda.amp.autocast(enabled=False):
            fft_combined = amp_attn.float() * torch.exp(1j * phase_shift.float())
            spatial_output = torch.fft.ifft2(fft_combined, dim=(-2, -1)).real # spatial_output은 float32 상태

        # 다시 original_dtype으로 변환
        spatial_output = spatial_output.to(original_dtype)


        # (4) Skip connection + combine conv
        sps = spatial_output + x  # 둘 다 original_dtype
        output = self.combine(sps)  # combine 파라미터 dtype == sps dtype

        return output

import torch
import torch.nn as nn
import torch.fft

__all__ = ('FrequencyExtractor')


class FrequencyExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(FrequencyExtractor, self).__init__()
        # 고주파와 저주파 처리 레이어 정의
        self.low_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.high_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # FFT 연산을 위한 데이터 형식 변환 (float32로 변환 후 FFT 수행)
        fft_input = x.to(dtype=torch.float32)
        
        # Fourier Transform 적용
        fft_x = torch.fft.fft2(fft_input, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft_x)

        # 저주파 성분 추출
        mid_h, mid_w = fft_input.shape[2] // 2, fft_input.shape[3] // 2
        low_freq = torch.real(fft_shifted[:, :, mid_h - 16:mid_h + 16, mid_w - 16:mid_w + 16])
        low_freq_padded = torch.nn.functional.pad(
            low_freq, (mid_w - 16, mid_w - 16, mid_h - 16, mid_h - 16)
        )

        # 고주파 성분 추출
        high_freq = torch.real(fft_shifted) - low_freq_padded

        # Conv 연산과 입력 데이터 형식 맞춤 (AMP 모드에서도 작동)
        low_freq_out = self.low_freq_conv(low_freq_padded.to(dtype=x.dtype))
        high_freq_out = self.high_freq_conv(high_freq.to(dtype=x.dtype))

        # 결과 결합 및 활성화
        combined = low_freq_out + high_freq_out
        return self.activation(combined)





'''
class FrequencyExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):

        super(FrequencyExtractor, self).__init__()
        # 고주파와 저주파 처리 레이어 정의
        self.low_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.high_freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        
        x = x.to(dtype=torch.float32)
        # Fourier Transform 적용
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft_x)

        # 저주파 성분 추출
        mid_h, mid_w = x.shape[2] // 2, x.shape[3] // 2
        low_freq = torch.real(fft_shifted[:, :, mid_h - 16 : mid_h + 16, mid_w - 16 : mid_w + 16])
        low_freq_padded = torch.nn.functional.pad(low_freq, (mid_w - 16, mid_w - 16, mid_h - 16, mid_h - 16))

        # 고주파 성분 추출
        high_freq = torch.real(fft_shifted) - low_freq_padded

        # 각각 처리
        low_freq_out = self.low_freq_conv(low_freq_padded)
        high_freq_out = self.high_freq_conv(high_freq)

        # 결과 결합 및 활성화
        combined = low_freq_out + high_freq_out
        return self.activation(combined)
'''
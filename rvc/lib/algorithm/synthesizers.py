import torch
from typing import Optional
from rvc.lib.algorithm.generators.hifigan_mrf import HiFiGANMRFGenerator
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from rvc.lib.algorithm.generators.hifigan import HiFiGANGenerator
from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator
from rvc.lib.algorithm.commons import slice_segments, rand_slice_segments
from rvc.lib.algorithm.residuals import ResidualCouplingBlock
from rvc.lib.algorithm.encoders import TextEncoder, PosteriorEncoder


class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model.

    Args:
        spec_channels (int): Number of channels in the spectrogram.
        segment_size (int): Size of the audio segment.
        inter_channels (int): Number of channels in the intermediate layers.
        hidden_channels (int): Number of channels in the hidden layers.
        filter_channels (int): Number of channels in the filter layers.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the encoder.
        kernel_size (int): Size of the convolution kernel.
        p_dropout (float): Dropout probability.
        resblock (str): Type of residual block.
        resblock_kernel_sizes (list): Kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for the residual blocks.
        upsample_rates (list): Upsampling rates for the decoder.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes for the upsampling layers.
        spk_embed_dim (int): Dimension of the speaker embedding.
        gin_channels (int): Number of channels in the global conditioning vector.
        sr (int): Sampling rate of the audio.
        use_f0 (bool): Whether to use F0 information.
        text_enc_hidden_dim (int): Hidden dimension for the text encoder.
        kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        randomized: bool = True,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.use_f0 = use_f0
        self.randomized = randomized

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        print(f"Using {vocoder} vocoder")
        if use_f0:
            if vocoder == "MRF HiFi-GAN":
                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=8,
                    checkpointing=checkpointing,
                )
            elif vocoder == "RefineGAN":
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                    checkpointing=checkpointing,
                )
            else:
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                    checkpointing=checkpointing,
                )
        else:
            if vocoder == "MRF HiFi-GAN":
                print("MRF HiFi-GAN does not support training without pitch guidance.")
                self.dec = None
            elif vocoder == "RefineGAN":
                print("RefineGAN does not support training without pitch guidance.")
                self.dec = None
            else:
                self.dec = HiFiGANGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

        self.retrieval_v2 = RetrievalAttentionV2(text_enc_hidden_dim)

    def _remove_weight_norm_from(self, module):
        for hook in module._forward_pre_hooks.values():
            if getattr(hook, "__class__", None).__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(module)

    def remove_weight_norm(self):
        for module in [self.dec, self.flow, self.enc_q]:
            self._remove_weight_norm_from(module)

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_lengths: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        gta: bool = False,
    ):

        if memory is not None:
            phone = self.retrieval_v2(phone, memory)

        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)

            dec_src = z
            if gta:  # decoder trains on the latents it will see at inference
                dec_src = self.flow(m_p, x_mask, g=g, reverse=True)

            # regular old training method using random slices
            if self.randomized:
                z_slice, ids_slice = rand_slice_segments(
                    dec_src, y_lengths, self.segment_size
                )
                if self.use_f0:
                    pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                    o = self.dec(z_slice, pitchf, g=g)
                else:
                    o = self.dec(z_slice, g=g)
                return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
            # future use for finetuning using the entire dataset each pass
            else:
                if self.use_f0:
                    o = self.dec(dec_src, pitchf, g=g)
                else:
                    o = self.dec(dec_src, g=g)
                return o, None, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = (
            self.dec(z * x_mask, nsff0, g=g)
            if self.use_f0
            else self.dec(z * x_mask, g=g)
        )

        return o, x_mask, (z, z_p, m_p, logs_p)


class RetrievalAttentionV2(torch.nn.Module):
    """Single-head retrieval attention over a bank of REAL target frames.
    NULL KEY: a learnable logit appended before softmax = a 'none of the
    above' candidate, so the module can honestly abstain (its share is
    discarded from the value mix). GATE: learned scalar, init -4
    (sigmoid ~0.018), controls how much retrieved content blends in."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.scale = dim**-0.5
        self.null_logit = torch.nn.Parameter(torch.zeros(1))
        self.gate = torch.nn.Parameter(torch.tensor(-4.0))
        self.last_gate = 0.0  # telemetry
        self.last_null_share = 0.0  # telemetry

    def forward(self, x, memory):
        # x: [B, T, D] content frames; memory: [B, M, D] bank of real frames
        logits = torch.matmul(x, memory.transpose(1, 2)) * self.scale  # [B,T,M]
        null = self.null_logit.expand(logits.shape[0], logits.shape[1], 1)
        w = torch.softmax(torch.cat([logits, null], dim=-1), dim=-1)  # [B,T,M+1]
        retrieved = torch.matmul(w[..., :-1], memory)  # null share discarded
        a = torch.sigmoid(self.gate)
        self.last_gate = float(a)
        self.last_null_share = float(w[..., -1].mean())
        return (1 - a) * x + a * retrieved

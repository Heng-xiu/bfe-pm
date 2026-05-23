"""bfe-pm model architecture: DualStreamBERT + quantile variant."""
import torch
import torch.nn as nn
from transformers import AutoModel

BERT_NAME = "trueto/medbert-kd-chinese"
N_STRUCT = 8


class DualStreamBERT(nn.Module):
    """Dual-stream surgical duration predictor.

    Text stream  : MedBERT-KD-Chinese CLS → Linear(768, 256) → GELU
    Struct stream: 8-dim features → wide MLP (192-64) with LayerNorm
    Fusion       : concat(256+64) → Linear(128) → Linear(1)
    """

    def __init__(
        self,
        bert_name: str = BERT_NAME,
        freeze_layers: int = 8,
        n_struct: int = N_STRUCT,
        wide_struct: bool = True,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        if wide_struct:
            self.struct_branch = nn.Sequential(
                nn.Linear(n_struct, 192), nn.GELU(), nn.LayerNorm(192), nn.Dropout(0.1),
                nn.Linear(192, 64), nn.GELU(),
            )
        else:
            self.struct_branch = nn.Sequential(
                nn.Linear(n_struct, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
            )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, struct_feats):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask
                        ).last_hidden_state[:, 0, :]
        text_repr   = self.text_proj(cls)
        struct_repr = self.struct_branch(struct_feats)
        return self.fusion(torch.cat([text_repr, struct_repr], dim=-1)).squeeze(-1)


class DualStreamBERTQuantile(DualStreamBERT):
    """Quantile-head variant for CQR (Q10 / Q50 / Q90)."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), **kwargs):
        super().__init__(**kwargs)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, len(quantiles)),
        )
        self.quantiles = quantiles

    def forward(self, input_ids, attention_mask, struct_feats):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask
                        ).last_hidden_state[:, 0, :]
        text_repr   = self.text_proj(cls)
        struct_repr = self.struct_branch(struct_feats)
        return self.fusion(torch.cat([text_repr, struct_repr], dim=-1))  # (B, 3)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles=(0.1, 0.5, 0.9)):
    """Joint pinball loss for multi-quantile output."""
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    target = target.unsqueeze(-1).expand_as(pred)
    diff = target - pred
    return torch.max(q * diff, (q - 1) * diff).mean()

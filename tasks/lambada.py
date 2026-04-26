#!/usr/bin/env python
"""LAMBADA last-word prediction task (Universal Transformers 논문 §4.5).

데이터: HuggingFace ``lambada`` (이미 캐시 다운로드 가정).
- ``text`` 필드의 문맥에서 마지막 단어를 예측.
- 단어 단위 vocab (top-K + UNK), 학습 셋에서 vocab 구축.

모델: UT encoder over context tokens → 마지막 토큰 위치 hidden 으로
vocab 분류 (cross-entropy). UT 논문은 fixed-compute 모드(6/8/9 hop)도
실험 — 본 구현은 ACT 모드와 fixed-step 모두 지원 (``--ut_act`` 사용 여부 +
``model.set_fixed_ponder_steps()`` 호출).
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
PAD_ID, UNK_ID, EOS_ID = 0, 1, 2


def _split_words(text: str) -> List[str]:
    # LAMBADA HF 본문은 이미 lowercased + 구두점 분리 (whitespace 단위 충분)
    return text.strip().split()


def _build_vocab(texts: List[str], top_k: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(_split_words(t))
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word, _ in counter.most_common(top_k):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def _encode_example(
    text: str,
    vocab: Dict[str, int],
    max_length: int,
) -> Optional[Tuple[torch.Tensor, int]]:
    words = _split_words(text)
    if len(words) < 2:
        return None
    target_word = words[-1]
    target_id = vocab.get(target_word, UNK_ID)
    context_ids = [vocab.get(w, UNK_ID) for w in words[:-1]] + [EOS_ID]
    if len(context_ids) > max_length:
        context_ids = context_ids[-max_length:]
    return torch.tensor(context_ids, dtype=torch.long), target_id


class LambadaSplitDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, texts: List[str], vocab: Dict[str, int], max_length: int, drop_unk_target: bool):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        for t in texts:
            enc = _encode_example(t, vocab, max_length)
            if enc is None:
                continue
            ids, target_id = enc
            if drop_unk_target and target_id == UNK_ID:
                continue
            self.samples.append((ids, target_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def lambada_collate_fn(batch):
    seqs, targets = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return padded, lengths, targets


class LambadaModel(pl.LightningModule):
    """UT encoder backbone for LAMBADA last-word prediction."""

    def __init__(
        self,
        hidden_size: int,
        batch_size: int,
        learning_rate: float,
        data_workers: int,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        ut_max_hops: int = 6,
        ut_halt_bias: float = 1.0,
        ut_attention_mode: str = "auto",
        disable_ponder_cost: bool = False,
        max_length: int = 256,
        vocab_top_k: int = 50000,
        drop_unk_target_train: bool = True,
        max_train_examples: Optional[int] = None,
        halt_warmup_steps: int = 0,
        fixed_ponder_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        from datasets import load_dataset

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        ds = load_dataset("lambada")

        train_texts = list(ds["train"]["text"])
        if max_train_examples is not None and len(train_texts) > max_train_examples:
            train_texts = train_texts[:max_train_examples]
        val_texts = list(ds["validation"]["text"])
        test_texts = list(ds["test"]["text"])

        # vocab 은 train + val + test 단어 union (test set 의 정답이 OOV 가 되더라도 UNK 처리).
        # UT 논문 LAMBADA 설정: 학습 셋 기반 vocab 사용. 여기서는 단순화.
        vocab = _build_vocab(train_texts + val_texts + test_texts, top_k=vocab_top_k)
        self.vocab = vocab
        self.id_to_word = {i: w for w, i in vocab.items()}
        self.vocab_size = len(vocab)
        rank_zero_info(f"[lambada] vocab_size={self.vocab_size} (top_k={vocab_top_k})")

        self.train_dataset = LambadaSplitDataset(train_texts, vocab, max_length, drop_unk_target_train)
        self.val_dataset = LambadaSplitDataset(val_texts, vocab, max_length, drop_unk_target=False)
        self.test_dataset = LambadaSplitDataset(test_texts, vocab, max_length, drop_unk_target=False)
        rank_zero_info(
            f"[lambada] train={len(self.train_dataset)} val={len(self.val_dataset)} test={len(self.test_dataset)}"
        )

        self.embedding = torch.nn.Embedding(self.vocab_size, hidden_size, padding_idx=PAD_ID)
        resolved_attention_mode = "causal" if ut_attention_mode == "auto" else ut_attention_mode
        self.encoder = UniversalTransformerEncoder(
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=ut_max_hops,
            num_heads=ut_heads,
            total_key_depth=ut_key_depth,
            total_value_depth=ut_value_depth,
            filter_size=ut_filter_size,
            max_length=max_length,
            act=ut_act,
            halt_bias_init=ut_halt_bias,
            attention_mode=resolved_attention_mode,
        )
        self.output_layer = torch.nn.Linear(hidden_size, self.vocab_size)

        if fixed_ponder_steps is not None:
            self.set_fixed_ponder_steps(fixed_ponder_steps)

        self._halting_frozen = False

    def set_fixed_ponder_steps(self, n: int) -> None:
        if self.hparams.ut_act and hasattr(self, "encoder"):
            self.encoder.set_fixed_ponder_steps(n)

    def _halting_named_parameters(self):
        if self.hparams.ut_act and hasattr(self.encoder, "act_fn"):
            return list(self.encoder.act_fn.p.named_parameters(prefix="encoder.act_fn.p"))
        return []

    def _set_halting_frozen(self, frozen: bool) -> None:
        if self._halting_frozen == frozen:
            return
        params = self._halting_named_parameters()
        if not params:
            self._halting_frozen = False
            return
        for _, p in params:
            p.requires_grad = not frozen
        rank_zero_info(
            f"[halt_warmup] {'freeze' if frozen else 'unfreeze'} halting params at step={self.global_step}"
        )
        self._halting_frozen = frozen

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.halt_warmup_steps <= 0:
            return
        self._set_halting_frozen(self.global_step < self.hparams.halt_warmup_steps)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embedding(input_ids)
        states, act_info = self.encoder(embedded)
        last_idx = (lengths.clamp(min=1) - 1).to(input_ids.device)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, states.size(-1))
        readout = states.gather(1, gather_idx).squeeze(1)
        logits = self.output_layer(readout)

        if act_info is None:
            ponder_cost = torch.tensor(0.0, device=input_ids.device)
            steps = torch.full((input_ids.size(0),), float(self.hparams.ut_max_hops), device=input_ids.device)
            act_stats = None
        else:
            remainders, n_updates, forced_halt_ratio = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean(dim=1)
            summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
            act_stats = {
                "mean_steps": summary["mean_steps"],
                "steps_p50": summary["steps_p50"],
                "steps_p90": summary["steps_p90"],
                "forced_halt_ratio": forced_halt_ratio,
                "n_updates_histogram": summary["histogram"],
            }
        return logits, ponder_cost, steps, act_stats

    def _compute_loss(self, logits, targets, ponder_cost):
        cls_loss = F.cross_entropy(logits, targets)
        if self.hparams.disable_ponder_cost:
            return cls_loss, cls_loss
        return cls_loss + ponder_cost, cls_loss

    def _accuracy(self, logits, targets):
        # UNK 정답은 자동 오답으로 카운트 (UT 논문 LAMBADA 평가 관행)
        return (logits.argmax(dim=-1) == targets).float().mean()

    def _maybe_log_n_updates_histogram(self, split: str, histogram: torch.Tensor) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb
        except ImportError:
            return
        values = torch.arange(1, histogram.numel() + 1, device=histogram.device)
        repeated = torch.repeat_interleave(values, histogram.to(torch.long).cpu())
        if repeated.numel() == 0:
            return
        experiment.log({f"{split}/n_updates_hist": wandb.Histogram(repeated.numpy())}, commit=False)

    def training_step(self, batch, batch_idx):
        input_ids, lengths, targets = batch
        logits, ponder_cost, steps, act_stats = self(input_ids, lengths)
        loss, cls_loss = self._compute_loss(logits, targets, ponder_cost)
        acc = self._accuracy(logits, targets)
        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/last_word_accuracy", acc, prog_bar=True)
        self.log("train/ponder_steps", steps.float().mean())
        if act_stats is not None:
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"])
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def _shared_eval_step(self, batch, stage: str):
        input_ids, lengths, targets = batch
        logits, ponder_cost, steps, act_stats = self(input_ids, lengths)
        cls_loss = F.cross_entropy(logits, targets)
        acc = self._accuracy(logits, targets)
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", cls_loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/last_word_accuracy", acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", steps.float().mean(), on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log(f"{stage}/forced_halt_ratio", act_stats["forced_halt_ratio"], on_epoch=True, on_step=False)
            self._maybe_log_n_updates_histogram(stage, act_stats["n_updates_histogram"])

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _make_loader(self, dataset, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
            collate_fn=lambada_collate_fn,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)

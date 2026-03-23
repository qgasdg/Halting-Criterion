#!/usr/bin/env python
"""String-based addition task with encoder-decoder seq2seq training."""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class StringAdditionVocab:
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    plus_token: str = "+"
    digits: str = "0123456789"

    def __post_init__(self):
        tokens = [self.pad_token, self.bos_token, self.eos_token, self.plus_token, *self.digits]
        stoi = {token: idx for idx, token in enumerate(tokens)}
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "stoi", stoi)
        object.__setattr__(self, "itos", {idx: token for token, idx in stoi.items()})
        object.__setattr__(self, "pad_id", stoi[self.pad_token])
        object.__setattr__(self, "bos_id", stoi[self.bos_token])
        object.__setattr__(self, "eos_id", stoi[self.eos_token])

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, token_ids: Sequence[int], stop_at_eos: bool = True, skip_special: bool = True) -> str:
        pieces: List[str] = []
        for token_id in token_ids:
            token = self.itos[int(token_id)]
            if stop_at_eos and token == self.eos_token:
                break
            if skip_special and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            pieces.append(token)
        return "".join(pieces)


class StringAdditionDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, sequence_length: int, max_digits: int, vocab: StringAdditionVocab | None = None):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for a+b style expressions.")
        if max_digits < 1:
            raise ValueError("max_digits must be at least 1.")
        self.sequence_length = sequence_length
        self.max_digits = max_digits
        self.vocab = vocab or StringAdditionVocab()

    def __iter__(self):
        while True:
            yield self._make_example()

    def _sample_number_string(self) -> str:
        digit_count = random.randint(1, self.max_digits)
        return "".join(random.choice(self.vocab.digits) for _ in range(digit_count))

    def _make_example(self) -> Dict[str, List[int]]:
        operand_count = self.sequence_length
        operands = [self._sample_number_string() for _ in range(operand_count)]
        expression = self.vocab.plus_token.join(operands)
        total = str(sum(int(operand) for operand in operands))

        src_tokens = list(expression) + [self.vocab.eos_token]
        tgt_tokens = [self.vocab.bos_token] + list(total)
        label_tokens = list(total) + [self.vocab.eos_token]
        return {
            "source_ids": self.vocab.encode(src_tokens),
            "decoder_input_ids": self.vocab.encode(tgt_tokens),
            "label_ids": self.vocab.encode(label_tokens),
        }


class FixedStringAdditionDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, sequence_length: int, max_digits: int, size: int, seed: int, vocab: StringAdditionVocab | None = None):
        if size <= 0:
            raise ValueError("size must be at least one.")
        generator = StringAdditionDataset(sequence_length=sequence_length, max_digits=max_digits, vocab=vocab)
        random_state = random.getstate()
        random.seed(seed)
        try:
            self.samples = [generator._make_example() for _ in range(size)]
        finally:
            random.setstate(random_state)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


class StringAdditionCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        source_ids = [torch.tensor(sample["source_ids"], dtype=torch.long) for sample in batch]
        decoder_input_ids = [torch.tensor(sample["decoder_input_ids"], dtype=torch.long) for sample in batch]
        label_ids = [torch.tensor(sample["label_ids"], dtype=torch.long) for sample in batch]

        return {
            "source_ids": torch.nn.utils.rnn.pad_sequence(source_ids, batch_first=True, padding_value=self.pad_id),
            "decoder_input_ids": torch.nn.utils.rnn.pad_sequence(
                decoder_input_ids, batch_first=True, padding_value=self.pad_id
            ),
            "label_ids": torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=self.pad_id),
        }


class StringAdditionModel(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        max_digits: int,
        hidden_size: int,
        batch_size: int,
        learning_rate: float,
        data_workers: int,
        val_size: int = 10000,
        eval_seed: int = 1234,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = StringAdditionVocab()
        self.dataset = StringAdditionDataset(sequence_length=sequence_length, max_digits=max_digits, vocab=self.vocab)
        self.val_dataset = FixedStringAdditionDataset(
            sequence_length=sequence_length,
            max_digits=max_digits,
            size=val_size,
            seed=eval_seed,
            vocab=self.vocab,
        )
        self.collate_fn = StringAdditionCollator(self.vocab.pad_id)

        self.embedding = nn.Embedding(self.vocab.size, hidden_size, padding_idx=self.vocab.pad_id)
        self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, self.vocab.size)

    def forward(self, source_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        source_emb = self.embedding(source_ids)
        _, encoder_hidden = self.encoder(source_emb)

        decoder_emb = self.embedding(decoder_input_ids)
        decoder_states, _ = self.decoder(decoder_emb, encoder_hidden)
        return self.output_layer(decoder_states)

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = logits.argmax(dim=-1)
        valid_mask = labels.ne(self.vocab.pad_id)
        token_correct = predictions.eq(labels) & valid_mask

        char_accuracy = token_correct.sum().float() / valid_mask.sum().clamp_min(1).float()
        sequence_correct = token_correct | ~valid_mask
        sequence_accuracy = sequence_correct.all(dim=1).float().mean()
        return predictions, char_accuracy, sequence_accuracy

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        logits = self(batch["source_ids"], batch["decoder_input_ids"])
        labels = batch["label_ids"]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab.size), labels.reshape(-1), ignore_index=self.vocab.pad_id)
        _, char_accuracy, sequence_accuracy = self._compute_metrics(logits, labels)
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/accuracy_char": char_accuracy,
                f"{stage}/accuracy_sequence": sequence_accuracy,
            },
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], _batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], _batch_idx: int):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collate_fn,
        )

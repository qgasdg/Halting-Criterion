#!/usr/bin/env python
"""bAbI QA task (Universal Transformers 논문 §4.4).

데이터 포맷 (FB tasks_1-20_v1-2):
- 각 줄: ``<line_num> <text>`` 또는 ``<line_num> <question>\\t<answer>\\t<support_indices>``
- ``line_num == 1`` 이면 새 story 시작.
- task 19 같은 다중 단어 정답은 콤마로 join 된 단일 토큰으로 취급 (단순화).

모델: Universal Transformer encoder 가 ``[s1] <SEP> [s2] <SEP> ... <SEP> [Q] <EOS>`` 형태의
시퀀스를 읽고 ``<EOS>`` 위치 hidden 으로 정답 단어를 분류한다.
"""

import glob
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, EOS_TOKEN]
PAD_ID, UNK_ID, SEP_ID, EOS_ID = 0, 1, 2, 3

DEFAULT_BABI_DIR = os.path.expanduser("~/.cache/huggingface/datasets/tasks_1-20_v1-2")
DEFAULT_VARIANT = "en-10k"  # UT 논문 §4.4: 10k 변형 사용

_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def _tokenize(text: str) -> List[str]:
    # 소문자 + 단어/숫자/구두점 단위 (UT 논문 단어 단위 vocab)
    return [t.lower() for t in _TOKEN_RE.findall(text) if not t.isspace()]


def _find_task_file(data_dir: str, variant: str, task_id: int, split: str) -> str:
    pattern = os.path.join(data_dir, variant, f"qa{task_id}_*_{split}.txt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No bAbI file found for pattern: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous bAbI file pattern matched multiple files: {matches}")
    return matches[0]


def _parse_babi_file(path: str) -> List[Tuple[List[List[str]], List[str], str]]:
    """Return list of (story_sentences, question_tokens, answer_string)."""
    examples: List[Tuple[List[List[str]], List[str], str]] = []
    story: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.split(" ", 1)
            line_num = int(sp[0])
            rest = sp[1] if len(sp) > 1 else ""
            if line_num == 1:
                story = []
            if "\t" in rest:
                parts = rest.split("\t")
                q_text, a_text = parts[0], parts[1]
                q_tokens = _tokenize(q_text)
                # 다중 토큰 정답(예: task 19 "s,e")은 단일 합성 토큰으로 처리
                answer = a_text.strip().lower().replace(" ", "")
                examples.append((list(story), q_tokens, answer))
            else:
                story.append(_tokenize(rest))
    return examples


def _build_vocab(examples: List[Tuple[List[List[str]], List[str], str]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for sents, question, answer in examples:
        for sent in sents:
            for tok in sent:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        for tok in question:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        if answer not in vocab:
            vocab[answer] = len(vocab)
    return vocab


def _encode_example(
    sents: List[List[str]],
    question: List[str],
    answer: str,
    vocab: Dict[str, int],
    max_length: int,
) -> Tuple[torch.Tensor, int]:
    ids: List[int] = []
    for sent in sents:
        ids.extend(vocab.get(w, UNK_ID) for w in sent)
        ids.append(SEP_ID)
    ids.extend(vocab.get(w, UNK_ID) for w in question)
    ids.append(EOS_ID)
    # 너무 길면 뒤쪽(질문 인접) 보존
    if len(ids) > max_length:
        ids = ids[-max_length:]
    answer_id = vocab.get(answer, UNK_ID)
    return torch.tensor(ids, dtype=torch.long), answer_id


class BabiDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(
        self,
        examples: List[Tuple[List[List[str]], List[str], str]],
        vocab: Dict[str, int],
        max_length: int,
    ):
        self.samples: List[Tuple[torch.Tensor, int]] = [
            _encode_example(s, q, a, vocab, max_length) for s, q, a in examples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def babi_collate_fn(batch):
    seqs, answers = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    answers = torch.tensor(answers, dtype=torch.long)
    return padded, lengths, answers


class BabiModel(pl.LightningModule):
    """UT encoder backbone for bAbI QA (per-task or joint training)."""

    def __init__(
        self,
        task_id: int,
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
        max_length: int = 512,
        data_dir: str = DEFAULT_BABI_DIR,
        variant: str = DEFAULT_VARIANT,
        val_fraction: float = 0.1,
        eval_seed: int = 1234,
        halt_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        train_examples = _parse_babi_file(_find_task_file(data_dir, variant, task_id, "train"))
        test_examples = _parse_babi_file(_find_task_file(data_dir, variant, task_id, "test"))

        # vocab 은 train + test 합쳐 구축 (단어 vocab 외 OOV 가 거의 없도록 — 표준 bAbI 관행)
        vocab = _build_vocab(train_examples + test_examples)
        self.vocab = vocab
        self.id_to_word = {i: w for w, i in vocab.items()}
        self.vocab_size = len(vocab)

        # 10% val split (UT 논문 관행)
        rng = random.Random(eval_seed)
        shuffled = list(train_examples)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_fraction))
        val_examples = shuffled[:val_size]
        train_only = shuffled[val_size:]

        self.train_dataset = BabiDataset(train_only, vocab, max_length)
        self.val_dataset = BabiDataset(val_examples, vocab, max_length)
        self.test_dataset = BabiDataset(test_examples, vocab, max_length)

        self.embedding = torch.nn.Embedding(self.vocab_size, hidden_size, padding_idx=PAD_ID)
        resolved_attention_mode = "full" if ut_attention_mode == "auto" else ut_attention_mode
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
        # readout: <EOS> 위치 (= length - 1, 단 padding 보정)
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

    def _compute_loss(self, logits, answers, ponder_cost):
        cls_loss = F.cross_entropy(logits, answers)
        if self.hparams.disable_ponder_cost:
            return cls_loss, cls_loss
        return cls_loss + ponder_cost, cls_loss

    def _accuracy(self, logits, answers):
        return (logits.argmax(dim=-1) == answers).float().mean()

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
        input_ids, lengths, answers = batch
        logits, ponder_cost, steps, act_stats = self(input_ids, lengths)
        loss, cls_loss = self._compute_loss(logits, answers, ponder_cost)
        acc = self._accuracy(logits, answers)
        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/accuracy", acc, prog_bar=True)
        self.log("train/ponder_steps", steps.float().mean())
        if act_stats is not None:
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"])
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def _shared_eval_step(self, batch, stage: str):
        input_ids, lengths, answers = batch
        logits, ponder_cost, steps, act_stats = self(input_ids, lengths)
        cls_loss = F.cross_entropy(logits, answers)
        acc = self._accuracy(logits, answers)
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", cls_loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/accuracy", acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
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
            collate_fn=babi_collate_fn,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)

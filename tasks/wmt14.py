#!/usr/bin/env python
"""WMT14 EN→DE machine translation (Universal Transformers 논문 §4.3).

Tokenization: SentencePiece BPE, EN+DE 합친 코퍼스로 학습한 공유 vocab.
Source/target: EN → DE.

평가 지표
=========
- Teacher-forced 토큰/시퀀스 정확도 (`{val,test}/token_accuracy`,
  `{val,test}/sequence_accuracy`)
- `--wmt14_eval_bleu` 플래그를 켜면 greedy decoding 결과로 sacrebleu corpus BLEU
  를 계산하여 `{val,test}/bleu` 로 로깅 (W&B 포함). 디코딩이 비싸기 때문에
  validation 은 `--wmt14_bleu_max_val_examples` 로 제한 (기본 500), test 는
  `--wmt14_bleu_max_test_examples` (기본 None=전체) 로 제어합니다.
"""

import os
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn

from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates
from tasks.string_addition import UniversalTransformerDecoder


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEFAULT_SP_CACHE = os.path.expanduser("~/.cache/halting-criterion/wmt14_sp")


def _train_sentencepiece(corpus_path: str, model_prefix: str, vocab_size: int) -> str:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        character_coverage=1.0,
        model_type="bpe",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )
    return model_prefix + ".model"


def _ensure_sp_model(
    train_pairs: List[Tuple[str, str]],
    sp_dir: str,
    vocab_size: int,
) -> str:
    os.makedirs(sp_dir, exist_ok=True)
    model_path = os.path.join(sp_dir, f"sp_{vocab_size}.model")
    if os.path.exists(model_path):
        return model_path

    corpus_file = os.path.join(sp_dir, f"corpus_{vocab_size}.txt")
    rank_zero_info(f"[wmt14] training SentencePiece (vocab={vocab_size}) on {len(train_pairs)} pairs → {corpus_file}")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for en, de in train_pairs:
            f.write(en.replace("\n", " ") + "\n")
            f.write(de.replace("\n", " ") + "\n")
    return _train_sentencepiece(corpus_file, os.path.join(sp_dir, f"sp_{vocab_size}"), vocab_size)


def _encode_pair(sp, en: str, de: str, max_length: int):
    src_ids = sp.encode(en, out_type=int)[: max_length - 1] + [EOS_ID]
    tgt_ids = sp.encode(de, out_type=int)[: max_length - 1] + [EOS_ID]
    dec_in = [BOS_ID] + tgt_ids[:-1]
    return (
        torch.tensor(src_ids, dtype=torch.long),
        torch.tensor(dec_in, dtype=torch.long),
        torch.tensor(tgt_ids, dtype=torch.long),
    )


class WMT14Dataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, pairs: List[Tuple[str, str]], sp, max_length: int):
        self.samples = []
        for en, de in pairs:
            src, dec_in, tgt = _encode_pair(sp, en, de, max_length)
            # 원본 DE 문자열을 함께 보관 → BLEU reference 로 사용 (탈토큰화 손실 방지).
            self.samples.append((src, dec_in, tgt, de))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def wmt14_collate_fn(batch):
    src, dec_in, tgt, ref_text = zip(*batch)
    return {
        "src_tokens": torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_ID),
        "decoder_input_tokens": torch.nn.utils.rnn.pad_sequence(dec_in, batch_first=True, padding_value=PAD_ID),
        "target_tokens": torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_ID),
        "reference_text": list(ref_text),
    }


class WMT14Model(pl.LightningModule):
    """UT encoder-decoder for WMT14 EN→DE translation."""

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
        sp_vocab_size: int = 32000,
        max_length: int = 128,
        max_train_pairs: Optional[int] = None,
        sp_cache_dir: str = DEFAULT_SP_CACHE,
        halt_warmup_steps: int = 0,
        eval_bleu: bool = False,
        decode_max_length: int = 128,
        bleu_max_val_examples: Optional[int] = 500,
        bleu_max_test_examples: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        from datasets import load_dataset

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        ds = load_dataset("wmt14", "de-en")

        # 학습 셋이 워낙 크기에 (~4.5M), max_train_pairs 로 제한 가능
        train_iter = ds["train"]
        if max_train_pairs is not None:
            train_iter = train_iter.select(range(min(max_train_pairs, len(train_iter))))
        train_pairs = [(ex["translation"]["en"], ex["translation"]["de"]) for ex in train_iter]
        val_pairs = [(ex["translation"]["en"], ex["translation"]["de"]) for ex in ds["validation"]]
        test_pairs = [(ex["translation"]["en"], ex["translation"]["de"]) for ex in ds["test"]]
        rank_zero_info(
            f"[wmt14] loaded train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}"
        )

        sp_path = _ensure_sp_model(train_pairs, sp_cache_dir, sp_vocab_size)
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)
        self.vocab_size = self.sp.get_piece_size()
        rank_zero_info(f"[wmt14] SP vocab_size={self.vocab_size}")

        self.train_dataset = WMT14Dataset(train_pairs, self.sp, max_length)
        self.val_dataset = WMT14Dataset(val_pairs, self.sp, max_length)
        self.test_dataset = WMT14Dataset(test_pairs, self.sp, max_length)

        self.embedding = torch.nn.Embedding(self.vocab_size, hidden_size, padding_idx=PAD_ID)
        self.output_layer = torch.nn.Linear(hidden_size, self.vocab_size)

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
        self.decoder = UniversalTransformerDecoder(
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=ut_max_hops,
            num_heads=ut_heads,
            total_key_depth=ut_key_depth,
            total_value_depth=ut_value_depth,
            filter_size=ut_filter_size,
            max_length=max_length,
            pad_id=PAD_ID,
        )

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

    def forward(self, src_tokens: torch.Tensor, decoder_input_tokens: torch.Tensor):
        src_embedded = self.embedding(src_tokens)
        decoder_embedded = self.embedding(decoder_input_tokens)
        memory, act_info = self.encoder(src_embedded)
        states = self.decoder(decoder_embedded, memory, decoder_input_tokens, src_tokens)
        logits = self.output_layer(states)

        if act_info is None:
            ponder_cost = torch.tensor(0.0, device=src_tokens.device)
            steps = torch.tensor(float(self.hparams.ut_max_hops), device=src_tokens.device)
            act_stats = None
        else:
            remainders, n_updates, forced_halt_ratio = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean()
            summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
            act_stats = {
                "mean_steps": summary["mean_steps"],
                "steps_p50": summary["steps_p50"],
                "steps_p90": summary["steps_p90"],
                "forced_halt_ratio": forced_halt_ratio,
                "n_updates_histogram": summary["histogram"],
            }
        return logits, ponder_cost, steps, act_stats

    def _compute_loss(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=PAD_ID,
        )

    def _accuracy(self, logits: torch.Tensor, target_tokens: torch.Tensor):
        predictions = logits.argmax(dim=-1)
        non_pad = target_tokens.ne(PAD_ID)
        char_acc = ((predictions == target_tokens) & non_pad).sum().float() / non_pad.sum().clamp_min(1)
        seq_acc = ((predictions == target_tokens) | ~non_pad).all(dim=1).float().mean()
        return char_acc, seq_acc

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
        logits, ponder_cost, steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        cls_loss = self._compute_loss(logits, batch["target_tokens"])
        loss = cls_loss if self.hparams.disable_ponder_cost else cls_loss + ponder_cost
        char_acc, seq_acc = self._accuracy(logits, batch["target_tokens"])

        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/ponder_steps", steps)
        self.log("train/token_accuracy", char_acc, prog_bar=True)
        self.log("train/sequence_accuracy", seq_acc, prog_bar=True)
        if act_stats is not None:
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"])
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def _shared_eval_step(self, batch, stage: str):
        logits, ponder_cost, steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        cls_loss = self._compute_loss(logits, batch["target_tokens"])
        char_acc, seq_acc = self._accuracy(logits, batch["target_tokens"])
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", cls_loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", steps, on_epoch=True, on_step=False)
        self.log(f"{stage}/token_accuracy", char_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/sequence_accuracy", seq_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log(f"{stage}/forced_halt_ratio", act_stats["forced_halt_ratio"], on_epoch=True, on_step=False)
            self._maybe_log_n_updates_histogram(stage, act_stats["n_updates_histogram"])

        if self.hparams.eval_bleu:
            self._maybe_collect_bleu(batch, stage)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test")

    # ------------------------------------------------------------------
    # BLEU evaluation (greedy decoding + sacrebleu corpus BLEU)
    # ------------------------------------------------------------------

    def _bleu_cap(self, stage: str) -> Optional[int]:
        return (
            self.hparams.bleu_max_val_examples if stage == "val" else self.hparams.bleu_max_test_examples
        )

    def _reset_bleu_buffer(self, stage: str) -> None:
        setattr(self, f"_{stage}_bleu_hyps", [])
        setattr(self, f"_{stage}_bleu_refs", [])
        setattr(self, f"_{stage}_bleu_count", 0)

    def _bleu_buffers(self, stage: str):
        return (
            getattr(self, f"_{stage}_bleu_hyps"),
            getattr(self, f"_{stage}_bleu_refs"),
        )

    def on_validation_epoch_start(self) -> None:
        if self.hparams.eval_bleu:
            self._reset_bleu_buffer("val")

    def on_test_epoch_start(self) -> None:
        if self.hparams.eval_bleu:
            self._reset_bleu_buffer("test")

    def on_validation_epoch_end(self) -> None:
        if self.hparams.eval_bleu:
            self._finalize_bleu("val")

    def on_test_epoch_end(self) -> None:
        if self.hparams.eval_bleu:
            self._finalize_bleu("test")

    def _maybe_collect_bleu(self, batch, stage: str) -> None:
        cap = self._bleu_cap(stage)
        count = getattr(self, f"_{stage}_bleu_count", 0)
        if cap is not None and count >= cap:
            return

        src_tokens = batch["src_tokens"]
        ref_text = batch.get("reference_text")
        if ref_text is None:
            # 안전 가드 — 데이터셋이 reference_text 를 제공하지 않는 경우.
            return

        remaining = src_tokens.size(0) if cap is None else max(0, cap - count)
        if remaining <= 0:
            return
        if remaining < src_tokens.size(0):
            src_tokens = src_tokens[:remaining]
            ref_text = ref_text[:remaining]

        decoded_ids = self._greedy_decode(src_tokens, self.hparams.decode_max_length)
        hyps = [self.sp.decode(ids) for ids in decoded_ids]

        hyp_buf, ref_buf = self._bleu_buffers(stage)
        hyp_buf.extend(hyps)
        ref_buf.extend(ref_text)
        setattr(self, f"_{stage}_bleu_count", count + len(hyps))

    @torch.no_grad()
    def _greedy_decode(self, src_tokens: torch.Tensor, max_length: int) -> List[List[int]]:
        """Autoregressive greedy decoding.

        Returns a list of lists of token IDs (one per batch item), with the
        leading BOS removed and truncated at the first EOS (EOS itself dropped).
        """
        was_training = self.training
        self.eval()
        try:
            batch_size = src_tokens.size(0)
            device = src_tokens.device

            src_embedded = self.embedding(src_tokens)
            memory, _ = self.encoder(src_embedded)

            dec_in = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_length):
                dec_emb = self.embedding(dec_in)
                states = self.decoder(dec_emb, memory, dec_in, src_tokens)
                next_logits = self.output_layer(states[:, -1, :])
                next_tokens = next_logits.argmax(dim=-1)
                # 이미 EOS 를 낸 시퀀스는 PAD 로 채워서 길이만 맞춤.
                next_tokens = torch.where(finished, torch.full_like(next_tokens, PAD_ID), next_tokens)
                dec_in = torch.cat([dec_in, next_tokens.unsqueeze(1)], dim=1)
                finished = finished | (next_tokens == EOS_ID)
                if bool(finished.all()):
                    break

            outputs: List[List[int]] = []
            for row in dec_in.tolist():
                # row[0] = BOS
                ids = row[1:]
                # 첫 EOS 위치까지로 자름.
                if EOS_ID in ids:
                    ids = ids[: ids.index(EOS_ID)]
                # 보호: PAD 제거 (finished 후 채워진 PAD).
                ids = [t for t in ids if t != PAD_ID]
                outputs.append(ids)
            return outputs
        finally:
            if was_training:
                self.train()

    def _finalize_bleu(self, stage: str) -> None:
        hyps, refs = self._bleu_buffers(stage)
        if not hyps:
            return
        try:
            import sacrebleu
        except ImportError:
            rank_zero_warn("[wmt14] sacrebleu 가 설치되지 않아 BLEU 를 건너뜁니다 (uv sync --extra nlp).")
            return
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        score = float(bleu.score)
        n_examples = len(hyps)
        rank_zero_info(
            f"[wmt14] {stage}/bleu={score:.2f} on {n_examples} examples (greedy decode, sacrebleu {bleu.signature if hasattr(bleu, 'signature') else ''})"
        )
        # Lightning 의 self.log 는 PL 로거(TB) + WandbLogger 에 모두 자동 dispatch.
        self.log(f"{stage}/bleu", score, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/bleu_n_examples", float(n_examples), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _make_loader(self, dataset, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
            collate_fn=wmt14_collate_fn,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)

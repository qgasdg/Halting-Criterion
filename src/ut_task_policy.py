from dataclasses import dataclass
from typing import Literal


AttentionMode = Literal["full", "causal"]
InputLayout = Literal["sequence", "grid_flat"]
EncoderVariant = Literal["standard"]
ReadoutHint = Literal["per_token", "mean_pool"]


@dataclass(frozen=True)
class UTTaskPolicy:
    attention_mode: AttentionMode
    input_layout: InputLayout
    encoder_variant: EncoderVariant = "standard"
    readout_hint: ReadoutHint = "per_token"


_TASK_POLICIES = {
    "addition": UTTaskPolicy(
        attention_mode="causal",
        input_layout="sequence",
        encoder_variant="standard",
        readout_hint="per_token",
    ),
    "parity": UTTaskPolicy(
        attention_mode="full",
        input_layout="sequence",
        encoder_variant="standard",
        readout_hint="mean_pool",
    ),
    "maze": UTTaskPolicy(
        attention_mode="full",
        input_layout="grid_flat",
        encoder_variant="standard",
        readout_hint="per_token",
    ),
    "sudoku": UTTaskPolicy(
        attention_mode="full",
        input_layout="grid_flat",
        encoder_variant="standard",
        readout_hint="per_token",
    ),
}


def get_ut_task_policy(task: str) -> UTTaskPolicy:
    try:
        return _TASK_POLICIES[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported UT task policy for task={task!r}.") from exc

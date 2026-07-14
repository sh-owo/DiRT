from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StateTrackingSample:
    prompt: str
    answer: str


def _apply_swap(state: dict[str, str], a: str, b: str) -> None:
    state[a], state[b] = state[b], state[a]


def _apply_move(state: dict[str, str], src: str, dst: str) -> None:
    state[dst] = state[src]


def _apply_copy(state: dict[str, str], src: str, dst: str) -> None:
    state[dst] = state[src]


def _apply_remove(state: dict[str, str], box: str) -> None:
    state[box] = "empty"


def generate_state_tracking_samples(
    n_samples: int,
    hop_choices: tuple[int, ...] = (2, 4, 6, 8),
    min_boxes: int = 3,
    max_boxes: int = 5,
    seed: int = 1234,
) -> list[StateTrackingSample]:
    rng = random.Random(seed)
    items = ["red", "blue", "green", "gold", "silver", "amber", "white", "black"]
    ops = ["swap", "move", "copy", "remove"]
    samples: list[StateTrackingSample] = []

    for _ in range(n_samples):
        n_boxes = rng.randint(min_boxes, max_boxes)
        box_names = [chr(ord("A") + i) for i in range(n_boxes)]
        state = {box: rng.choice(items) for box in box_names}

        init_sentence = " ".join([f"Box {b} has {state[b]}." for b in box_names])
        n_hops = rng.choice(hop_choices)
        steps: list[str] = []

        for _hop in range(n_hops):
            op = rng.choice(ops)
            if op == "swap":
                a, b = rng.sample(box_names, 2)
                _apply_swap(state, a, b)
                steps.append(f"Swap {a} and {b}.")
            elif op == "move":
                src, dst = rng.sample(box_names, 2)
                _apply_move(state, src, dst)
                steps.append(f"Move from {src} to {dst}.")
            elif op == "copy":
                src, dst = rng.sample(box_names, 2)
                _apply_copy(state, src, dst)
                steps.append(f"Copy from {src} to {dst}.")
            else:
                target = rng.choice(box_names)
                _apply_remove(state, target)
                steps.append(f"Remove from {target}.")

        answer_box = rng.choice(box_names)
        question = f"Where is the content of box {answer_box}?"
        prompt = " ".join([init_sentence] + steps + [question])
        answer = state[answer_box]
        samples.append(StateTrackingSample(prompt=prompt, answer=answer))

    return samples

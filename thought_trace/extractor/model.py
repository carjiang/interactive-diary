"""
BERT + CRF model for NER-based diary event extraction.

Components:
  - NERDataset: torch Dataset — WordPiece tokenisation + word-level alignment.
  - CRF: linear-chain conditional random field (Viterbi decode).
  - BertCRFForNER: BertModel → word-pool → dropout → linear → CRF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizerFast

from thought_trace.extractor.schema import (
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    NERConfig,
    NERSample,
)

IGNORE_LABEL_ID: int = -100


class NERDataset(Dataset):
    """Converts pre-tokenised NERSamples into BERT-ready tensors.

    Returns sub-token-level inputs for BERT and word-level labels for
    the classifier/CRF.  The model pools BERT hidden states to word-level
    using ``word_starts`` before the classification head, so the CRF
    operates on a dense, contiguous word sequence.
    """

    def __init__(
        self,
        samples: list[NERSample],
        tokenizer: BertTokenizerFast,
        max_seq_length: int = 512,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.tokens,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_word_ids = encoding.word_ids(batch_index=0)

        first_subtoken: dict[int, int] = {}
        for pos, wid in enumerate(raw_word_ids):
            if wid is not None:
                first_subtoken.setdefault(wid, pos)

        num_words = len(first_subtoken)

        word_starts = torch.zeros(self.max_seq_length, dtype=torch.long)
        for wid, pos in first_subtoken.items():
            word_starts[wid] = pos

        word_labels = torch.full(
            (self.max_seq_length,), IGNORE_LABEL_ID, dtype=torch.long,
        )
        word_labels[:num_words] = torch.tensor(
            [LABEL_TO_ID[tag] for tag in sample.bio_tags[:num_words]],
            dtype=torch.long,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "word_starts": word_starts,
            "labels": word_labels,
            "num_words": torch.tensor(num_words, dtype=torch.long),
        }


class CRF(nn.Module):
    """Linear-chain Conditional Random Field.

    Operates on (batch, seq_len, num_tags) emission tensors.
    ``forward()`` returns negative log-likelihood loss;
    ``decode()`` returns best tag sequences via Viterbi.

    Hard BIO constraints are enforced via immutable registered buffers
    (not learnable parameters), so they cannot be eroded by the optimizer.
    Structurally invalid transitions (e.g. I-ACTION after B-PARTICIPANT)
    receive -10000 and never appear in decoded sequences.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._register_constraint_masks()

    def _register_constraint_masks(self) -> None:
        """Build immutable masks that enforce BIO transition legality.

        Registered as buffers so they move with ``.to(device)`` / ``.cuda()``
        but are never updated by the optimizer.
        """
        NEG_INF = -10_000.0
        trans_mask = torch.zeros(self.num_tags, self.num_tags)
        start_mask = torch.zeros(self.num_tags)

        for curr_id in range(self.num_tags):
            curr_label = ID_TO_LABEL[curr_id]
            if not curr_label.startswith("I-"):
                continue
            entity = curr_label[2:]
            start_mask[curr_id] = NEG_INF
            for prev_id in range(self.num_tags):
                prev_label = ID_TO_LABEL[prev_id]
                if prev_label not in (f"B-{entity}", f"I-{entity}"):
                    trans_mask[prev_id, curr_id] = NEG_INF

        self.register_buffer("transition_mask", trans_mask)
        self.register_buffer("start_mask", start_mask)

    @property
    def constrained_transitions(self) -> torch.Tensor:
        """Learned transitions + immutable BIO-legality mask."""
        return self.transitions + self.transition_mask

    @property
    def constrained_start(self) -> torch.Tensor:
        """Learned start transitions + immutable BIO-legality mask."""
        return self.start_transitions + self.start_mask

    def _step(self, score: torch.Tensor, emission: torch.Tensor) -> torch.Tensor:
        """Candidate scores for one timestep: (batch, prev_tag, curr_tag)."""
        return score.unsqueeze(2) + self.constrained_transitions + emission.unsqueeze(1)

    def forward(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_tags)
            labels:    (batch, seq_len) — padding positions are clamped to 0
                       internally and excluded via mask.
            mask:      (batch, seq_len) — 1 for real tokens, 0 for padding.
        """
        gold_score = self._score_sentence(emissions, labels, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi decoding — best tag-id sequence per batch element."""
        batch_size, seq_len, _ = emissions.shape

        score = self.constrained_start + emissions[:, 0]
        history: list[torch.Tensor] = []

        for i in range(1, seq_len):
            candidates = self._step(score, emissions[:, i])
            next_score, indices = candidates.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_lengths = mask.long().sum(dim=1)

        best_tags_list: list[list[int]] = []
        for b in range(batch_size):
            best_last = score[b].argmax().item()
            length = seq_lengths[b].item()
            best_tags = [best_last]

            for hist in reversed(history[: length - 1]):
                best_last = hist[b][best_last].item()
                best_tags.append(best_last)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = emissions.shape
        tags = tags.clamp(min=0)

        score = self.constrained_start[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            curr, prev = tags[:, i], tags[:, i - 1]
            emit = emissions[:, i].gather(1, curr.unsqueeze(1)).squeeze(1)
            trans = self.constrained_transitions[prev, curr]
            score += (emit + trans) * mask[:, i]

        last_idx = mask.long().sum(dim=1) - 1
        last_tag = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tag]
        return score

    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = emissions.shape

        score = self.constrained_start + emissions[:, 0]

        for i in range(1, seq_len):
            next_score = torch.logsumexp(self._step(score, emissions[:, i]), dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)


class BertCRFForNER(nn.Module):
    """BertModel -> word-pool -> dropout -> linear -> CRF.

    BERT produces sub-token hidden states; ``_get_word_emissions`` pools
    them to word-level (first sub-token per word) before the classification
    head so the CRF operates on a dense, contiguous word sequence.
    """

    def __init__(self, config: NERConfig) -> None:
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, len(BIO_LABELS))
        self.crf: CRF | None = CRF(len(BIO_LABELS)) if config.use_crf else None

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self._freeze_layers()

    def _freeze_layers(self) -> None:
        if self.config.freeze_bert_layers <= 0:
            return
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[: self.config.freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def _get_word_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        num_words: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BERT, pool to word-level, and classify.

        Returns:
            emissions:  (batch, max_words, num_labels)
            word_mask:  (batch, max_words) float — 1 for real words, 0 for padding.
        """
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state

        max_words = int(num_words.max().item())
        hidden_size = hidden.size(2)

        gather_idx = word_starts[:, :max_words].unsqueeze(-1).expand(-1, -1, hidden_size)
        word_hidden = hidden.gather(1, gather_idx)

        emissions = self.classifier(self.dropout(word_hidden))

        word_mask = (
            torch.arange(max_words, device=num_words.device).unsqueeze(0)
            < num_words.unsqueeze(1)
        ).float()

        return emissions, word_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        num_words: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with ``"emissions"`` and (if labels given) ``"loss"``."""
        emissions, word_mask = self._get_word_emissions(
            input_ids, attention_mask, word_starts, num_words,
        )

        result: dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is not None:
            word_labels = labels[:, : emissions.size(1)]
            if self.crf is not None:
                result["loss"] = self.crf(emissions, word_labels, word_mask)
            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
                result["loss"] = loss_fn(
                    emissions.reshape(-1, len(BIO_LABELS)),
                    word_labels.reshape(-1),
                )

        return result

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        num_words: torch.Tensor,
    ) -> list[list[int]]:
        """Best word-level tag sequences.

        Automatically sets eval mode (disabling dropout) and restores the
        original training state afterward.  Uses ``torch.inference_mode``
        for maximum inference performance.
        """
        tag_seqs, _, _ = self.decode_with_emissions(
            input_ids, attention_mask, word_starts, num_words,
        )
        return tag_seqs

    def decode_with_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        num_words: torch.Tensor,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
        """Decode and return emissions in a single BERT forward pass.

        Returns:
            tag_seqs:  Best tag-id sequence per batch element.
            emissions: (batch, max_words, num_labels) logits.
            word_mask: (batch, max_words) float mask.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                emissions, word_mask = self._get_word_emissions(
                    input_ids, attention_mask, word_starts, num_words,
                )
                if self.crf is not None:
                    tag_seqs = self.crf.decode(emissions, word_mask)
                else:
                    all_ids = emissions.argmax(dim=-1).tolist()
                    tag_seqs = [ids[:n] for ids, n in zip(all_ids, num_words.tolist())]
                return tag_seqs, emissions, word_mask
        finally:
            self.train(was_training)

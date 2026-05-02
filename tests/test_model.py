"""Tests for model.py: CRF constraints, BIO validity, train/serve parity,
freeze layers, non-CRF path, and overfit sanity check.

These instantiate real BERT weights so they're slower (~10-20s) but
exercise the actual model code end-to-end.
"""

from __future__ import annotations

import torch
import pytest

from transformers import BertTokenizerFast

from thought_trace.extractor.schema import (
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    NERConfig,
    NERSample,
)
from thought_trace.extractor.model import BertCRFForNER, CRF, NERDataset, IGNORE_LABEL_ID
from thought_trace.extractor.inference import _prepare_batch


NUM_TAGS = len(BIO_LABELS)


def _is_valid_bio(tag_ids: list[int]) -> bool:
    """Return True if the tag-id sequence is a legal BIO sequence."""
    for i, tid in enumerate(tag_ids):
        label = ID_TO_LABEL.get(tid, "O")
        if not label.startswith("I-"):
            continue
        entity = label[2:]
        if i == 0:
            return False
        prev_label = ID_TO_LABEL.get(tag_ids[i - 1], "O")
        if prev_label not in (f"B-{entity}", f"I-{entity}"):
            return False
    return True


# ---------------------------------------------------------------------------
# CRF constraint masks
# ---------------------------------------------------------------------------

class TestCRFConstraints:
    @pytest.fixture
    def crf(self):
        return CRF(NUM_TAGS)

    def test_i_tags_blocked_at_start(self, crf):
        """Every I-* tag should have -10000 in the start mask."""
        for tid in range(NUM_TAGS):
            label = ID_TO_LABEL[tid]
            if label.startswith("I-"):
                assert crf.start_mask[tid].item() == -10_000.0, (
                    f"{label} (id={tid}) should be blocked at start"
                )
            else:
                assert crf.start_mask[tid].item() == 0.0

    def test_illegal_transitions_blocked(self, crf):
        """I-X can only follow B-X or I-X. All other prev -> I-X must be -10000."""
        for curr_id in range(NUM_TAGS):
            curr_label = ID_TO_LABEL[curr_id]
            if not curr_label.startswith("I-"):
                continue
            entity = curr_label[2:]
            for prev_id in range(NUM_TAGS):
                prev_label = ID_TO_LABEL[prev_id]
                mask_val = crf.transition_mask[prev_id, curr_id].item()
                if prev_label in (f"B-{entity}", f"I-{entity}"):
                    assert mask_val == 0.0, (
                        f"{prev_label} -> {curr_label} should be allowed"
                    )
                else:
                    assert mask_val == -10_000.0, (
                        f"{prev_label} -> {curr_label} should be blocked"
                    )

    def test_o_and_b_tags_unconstrained_at_start(self, crf):
        """O and B-* tags should have 0.0 in the start mask (learnable)."""
        for tid in range(NUM_TAGS):
            label = ID_TO_LABEL[tid]
            if label == "O" or label.startswith("B-"):
                assert crf.start_mask[tid].item() == 0.0

    def test_constraint_buffers_not_parameters(self, crf):
        """Constraint masks must be buffers, not learnable parameters."""
        param_names = {n for n, _ in crf.named_parameters()}
        assert "transition_mask" not in param_names
        assert "start_mask" not in param_names
        buffer_names = {n for n, _ in crf.named_buffers()}
        assert "transition_mask" in buffer_names
        assert "start_mask" in buffer_names


# ---------------------------------------------------------------------------
# CRF Viterbi always produces valid BIO
# ---------------------------------------------------------------------------

class TestCRFViterbiValidity:
    def test_random_emissions_produce_valid_bio(self):
        """Decode 50 random emission tensors; every output must be valid BIO."""
        crf = CRF(NUM_TAGS)
        crf.eval()

        for _ in range(50):
            seq_len = torch.randint(1, 20, (1,)).item()
            emissions = torch.randn(1, seq_len, NUM_TAGS)
            mask = torch.ones(1, seq_len)
            with torch.inference_mode():
                decoded = crf.decode(emissions, mask)
            assert len(decoded) == 1
            assert len(decoded[0]) == seq_len
            assert _is_valid_bio(decoded[0]), (
                f"Invalid BIO: {[ID_TO_LABEL[t] for t in decoded[0]]}"
            )

    def test_batch_decode_valid_bio(self):
        """Batch of 4 with different lengths, all must be valid BIO."""
        crf = CRF(NUM_TAGS)
        crf.eval()

        max_len = 12
        batch_size = 4
        emissions = torch.randn(batch_size, max_len, NUM_TAGS)
        lengths = [12, 8, 5, 3]
        mask = torch.zeros(batch_size, max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0

        with torch.inference_mode():
            decoded = crf.decode(emissions, mask)

        for i, (tags, length) in enumerate(zip(decoded, lengths)):
            assert len(tags) == length, f"Batch {i}: expected {length} tags, got {len(tags)}"
            assert _is_valid_bio(tags), (
                f"Batch {i}: invalid BIO: {[ID_TO_LABEL[t] for t in tags]}"
            )


# ---------------------------------------------------------------------------
# CRF loss sanity: forward_score >= gold_score (NLL >= 0 on average)
# ---------------------------------------------------------------------------

class TestCRFLoss:
    def test_loss_is_nonnegative_on_valid_sequence(self):
        """NLL loss from a valid gold sequence should be non-negative."""
        crf = CRF(NUM_TAGS)
        emissions = torch.randn(1, 5, NUM_TAGS)
        labels = torch.tensor([[LABEL_TO_ID["O"]] * 5])
        mask = torch.ones(1, 5)
        loss = crf(emissions, labels, mask)
        assert loss.item() >= 0.0, f"CRF loss should be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Train/serve tensor parity
# ---------------------------------------------------------------------------

class TestTrainServeParity:
    """NERDataset (train) and _prepare_batch (inference) must produce
    identical word_starts and num_words for the same tokenized input."""

    @pytest.fixture
    def tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    def _check_parity(self, tokenizer, tokens, bio_tags, max_seq_length=64):
        sample = NERSample(tokens=tokens, bio_tags=bio_tags)
        ds = NERDataset([sample], tokenizer, max_seq_length=max_seq_length)
        train_item = ds[0]

        infer_batch = _prepare_batch(tokens, tokenizer, max_seq_length, torch.device("cpu"))

        train_num = train_item["num_words"].item()
        infer_num = infer_batch["num_words"].item()
        assert train_num == infer_num, (
            f"num_words mismatch: train={train_num}, infer={infer_num}"
        )

        train_ws = train_item["word_starts"][:train_num]
        infer_ws = infer_batch["word_starts"].squeeze(0)[:infer_num]
        assert torch.equal(train_ws, infer_ws), (
            f"word_starts mismatch:\n  train={train_ws.tolist()}\n  infer={infer_ws.tolist()}"
        )

        assert torch.equal(
            train_item["input_ids"],
            infer_batch["input_ids"].squeeze(0),
        ), "input_ids mismatch"

        assert torch.equal(
            train_item["attention_mask"],
            infer_batch["attention_mask"].squeeze(0),
        ), "attention_mask mismatch"

    def test_simple_sentence(self, tokenizer):
        self._check_parity(
            tokenizer,
            ["I", "told", "Sarah", "about", "the", "meeting"],
            ["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT", "O", "B-CONTENT", "I-CONTENT"],
        )

    def test_subword_heavy_sentence(self, tokenizer):
        """Words that split into many sub-tokens (e.g. 'unbelievable')."""
        self._check_parity(
            tokenizer,
            ["I", "found", "it", "unbelievable"],
            ["B-PARTICIPANT", "B-ACTION", "O", "B-CONTENT"],
        )

    def test_single_token(self, tokenizer):
        self._check_parity(tokenizer, ["hello"], ["O"])


# ---------------------------------------------------------------------------
# Non-CRF code path
# ---------------------------------------------------------------------------

class TestNonCRFPath:
    @pytest.fixture
    def model_no_crf(self):
        cfg = NERConfig(use_crf=False, max_seq_length=32)
        m = BertCRFForNER(cfg)
        m.eval()
        return m

    def test_decode_returns_valid_shapes(self, model_no_crf):
        input_ids = torch.randint(0, 1000, (1, 32))
        mask = torch.ones(1, 32, dtype=torch.long)
        ws = torch.arange(5).unsqueeze(0)
        nw = torch.tensor([5])

        tags = model_no_crf.decode(input_ids, mask, ws, nw)
        assert len(tags) == 1
        assert len(tags[0]) == 5

    def test_decode_with_emissions_returns_all(self, model_no_crf):
        input_ids = torch.randint(0, 1000, (1, 32))
        mask = torch.ones(1, 32, dtype=torch.long)
        ws = torch.arange(5).unsqueeze(0)
        nw = torch.tensor([5])

        tags, emissions, word_mask = model_no_crf.decode_with_emissions(
            input_ids, mask, ws, nw,
        )
        assert emissions.shape == (1, 5, NUM_TAGS)
        assert word_mask.shape == (1, 5)
        assert tags == model_no_crf.decode(input_ids, mask, ws, nw)

    def test_forward_with_labels_produces_loss(self, model_no_crf):
        input_ids = torch.randint(0, 1000, (1, 32))
        mask = torch.ones(1, 32, dtype=torch.long)
        ws = torch.arange(5).unsqueeze(0)
        nw = torch.tensor([5])
        labels = torch.zeros(1, 32, dtype=torch.long)

        model_no_crf.train()
        result = model_no_crf(input_ids, mask, ws, nw, labels)
        assert "loss" in result
        assert result["loss"].requires_grad


# ---------------------------------------------------------------------------
# Freeze layers
# ---------------------------------------------------------------------------

class TestFreezeLayers:
    def test_freeze_zero_layers(self):
        cfg = NERConfig(freeze_bert_layers=0, max_seq_length=32)
        model = BertCRFForNER(cfg)
        for p in model.bert.embeddings.parameters():
            assert p.requires_grad is True

    def test_freeze_two_layers(self):
        cfg = NERConfig(freeze_bert_layers=2, max_seq_length=32)
        model = BertCRFForNER(cfg)

        for p in model.bert.embeddings.parameters():
            assert p.requires_grad is False, "Embeddings should be frozen"
        for p in model.bert.encoder.layer[0].parameters():
            assert p.requires_grad is False, "Layer 0 should be frozen"
        for p in model.bert.encoder.layer[1].parameters():
            assert p.requires_grad is False, "Layer 1 should be frozen"
        for p in model.bert.encoder.layer[2].parameters():
            assert p.requires_grad is True, "Layer 2 should NOT be frozen"

        for p in model.classifier.parameters():
            assert p.requires_grad is True, "Classifier should never be frozen"


# ---------------------------------------------------------------------------
# Overfit sanity: loss decreases over N steps on one sample
# ---------------------------------------------------------------------------

class TestOverfitSanity:
    def test_loss_decreases_on_single_fixture(self):
        """Train on one sample for 20 steps. The best loss in the second half
        must be below the initial loss -- proving the model is learning."""
        cfg = NERConfig(
            max_seq_length=32,
            learning_rate=2e-4,
            use_crf=True,
        )
        tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)

        sample = NERSample(
            tokens=["I", "told", "Sarah", "about", "the", "meeting"],
            bio_tags=["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT", "O", "B-CONTENT", "I-CONTENT"],
        )
        ds = NERDataset([sample], tokenizer, max_seq_length=cfg.max_seq_length)
        item = ds[0]

        model = BertCRFForNER(cfg)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)
        word_starts = item["word_starts"].unsqueeze(0)
        num_words = item["num_words"].unsqueeze(0)
        labels = item["labels"].unsqueeze(0)

        n_steps = 20
        losses = []
        for _ in range(n_steps):
            optimizer.zero_grad()
            out = model(input_ids, attention_mask, word_starts, num_words, labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        best_second_half = min(losses[n_steps // 2:])
        assert best_second_half < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, "
            f"best_second_half={best_second_half:.4f}\n"
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )


# ---------------------------------------------------------------------------
# NERDataset edge cases
# ---------------------------------------------------------------------------

class TestNERDataset:
    @pytest.fixture
    def tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    def test_labels_padded_with_ignore(self, tokenizer):
        sample = NERSample(tokens=["hello"], bio_tags=["O"])
        ds = NERDataset([sample], tokenizer, max_seq_length=32)
        item = ds[0]

        assert item["labels"][0].item() == LABEL_TO_ID["O"]
        assert (item["labels"][1:] == IGNORE_LABEL_ID).all()

    def test_num_words_matches_tokens(self, tokenizer):
        tokens = ["I", "told", "Sarah"]
        tags = ["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT"]
        sample = NERSample(tokens=tokens, bio_tags=tags)
        ds = NERDataset([sample], tokenizer, max_seq_length=64)
        item = ds[0]

        assert item["num_words"].item() == 3

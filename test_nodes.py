"""Run from the ComfyUI environment: python test_nodes.py --cpu."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).parent))
import torch
from nodes import TextGenerateMultiImage, MultiImageClip, collect_images, image_payloads, final_answer
from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer


class Clip:
    def __init__(self):
        self.tokenizer = Qwen35ImageTokenizer(model_type="qwen35_27b")

    def tokenize(self, prompt, **kwargs):
        self.prompt = prompt
        return self.tokenizer.tokenize_with_weights(prompt, **kwargs)

    def generate(self, tokens, **kwargs):
        self.tokens, self.sampling = tokens, kwargs
        return [1]

    def decode(self, ids, skip_special_tokens=True):
        text = getattr(self, "output", "generated text")
        if skip_special_tokens:
            return text.replace("<think>", "").replace("</think>", "")
        return text


class MultiImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = Clip()

    def test_sparse_inputs_batches_and_different_sizes(self):
        inputs = {"image_2": torch.zeros(1, 64, 96, 3),
                  "image_8": torch.ones(2, 96, 64, 3)}
        images, labels = collect_images(inputs)
        tokens = MultiImageClip(self.clip, images).tokenize("Describe.")
        self.assertEqual(len(list(image_payloads(tokens))), 3)
        self.assertEqual(labels, ["<Picture 1> = image_2",
                                 "<Picture 2> = image_8, batch item 1",
                                 "<Picture 3> = image_8, batch item 2"])

    def test_all_eight_sockets(self):
        images, _ = collect_images({f"image_{i}": torch.zeros(1, 32, 32, 3)
                                    for i in range(1, 9)})
        tokens = MultiImageClip(self.clip, images).tokenize("Describe.")
        self.assertEqual(len(list(image_payloads(tokens))), 8)

    def test_custom_template_missing_placeholders_fails(self):
        images, _ = collect_images({"image_1": torch.zeros(1, 32, 32, 3)})
        with self.assertRaisesRegex(ValueError, "did not attach"):
            MultiImageClip(self.clip, images).tokenize(
                "<|im_start|>user\nMissing image placeholder<|im_end|>"
            )

    def test_text_only_and_sampling_passthrough(self):
        sampling = {"sampling_mode": "on", "seed": 123, "temperature": 0.61,
                    "top_k": 20, "top_p": 0.8, "min_p": 0.1,
                    "repetition_penalty": 1.1, "presence_penalty": 1.5}
        result = TextGenerateMultiImage.execute(self.clip, "Exact prompt", 75, sampling)
        self.assertEqual(result.args[0], "generated text")
        self.assertEqual(self.clip.prompt, "Exact prompt")
        self.assertEqual(list(image_payloads(self.clip.tokens)), [])
        self.assertEqual(self.clip.sampling, dict(
            do_sample=True, max_length=75,
            **{k: v for k, v in sampling.items() if k != "sampling_mode"}
        ))

    def test_generation_includes_reference_index(self):
        result = TextGenerateMultiImage.execute(
            self.clip, "Original request", 50, {"sampling_mode": "off"},
            image_1=torch.zeros(1, 32, 32, 3), image_4=torch.ones(1, 48, 32, 3),
        )
        self.assertEqual(result.args[0], "generated text")
        self.assertIn("<Picture 2> = image_4", self.clip.prompt)
        self.assertEqual(len(list(image_payloads(self.clip.tokens))), 2)
        self.assertFalse(self.clip.sampling["do_sample"])

    def test_invalid_image_is_rejected(self):
        for shape in [(0, 32, 32, 3), (1, 32, 32, 4), (32, 32, 3)]:
            with self.assertRaisesRegex(ValueError, "image_1"):
                collect_images({"image_1": torch.zeros(*shape)})

    def test_thinking_generation_preserves_sampling_images_and_dialogue(self):
        clip = Clip()
        expected = '<Subject 1> (S1) says: <d>[Portuguese] olha isso</d>'
        clip.output = '<think>private planning</think>\n' + expected + '<|im_end|>'
        result = TextGenerateMultiImage.execute(
            clip, "Request", 100, {"sampling_mode": "off"}, thinking=True,
            image_1=torch.zeros(1, 32, 32, 3),
        )
        self.assertEqual(result.args[0], expected)
        self.assertEqual(len(list(image_payloads(clip.tokens))), 1)
        self.assertFalse(clip.sampling["do_sample"])

    def test_incomplete_thinking_never_reaches_output(self):
        clip = Clip()
        clip.output = '<think>unfinished planning'
        with self.assertRaisesRegex(ValueError, "No completed thinking boundary"):
            TextGenerateMultiImage.execute(
                clip, "Request", 10, {"sampling_mode": "off"}, thinking=True,
            )


class FinalAnswerTests(unittest.TestCase):
    def test_complete_and_prefilled_reasoning(self):
        for raw in ['<think>planning</think>\nFinal', 'planning</think>\nFinal',
                    '<think>\n</think>Final<|im_end|><|endoftext|>']:
            with self.subTest(raw=raw):
                self.assertEqual(final_answer(raw, thinking=True), 'Final')

    def test_unexpected_reasoning_when_disabled(self):
        self.assertEqual(final_answer('<think>planning</think>Final'), 'Final')

    def test_missing_empty_and_truncated_final_are_errors(self):
        for raw in ['<think>unfinished', 'unmarked planning', '<think>x</think>',
                    '<think>x</think><|im_end|>', '<think>x</think>Final<think>unfinished']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                final_answer(raw, thinking=True)

    def test_additional_reasoning_block(self):
        self.assertEqual(final_answer('<think>x</think>A<think>y</think>B'), 'AB')


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])

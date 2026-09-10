"""Multiple separate visual references for ComfyUI's native text generator."""

import re

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_textgen import TextGenerate


def final_answer(text, thinking=False):
    """Extract Qwen's final channel before special-token decoding loses it."""
    if "</think>" in text:
        # The opening tag may be part of a custom template rather than output.
        text = text.split("</think>", 1)[1]
        while "<think>" in text:
            start = text.index("<think>")
            end = text.find("</think>", start)
            if end < 0:
                raise ValueError("Thinking output is incomplete. Increase max_length and retry.")
            text = text[:start] + text[end + len("</think>"):]
        if "</think>" in text:
            raise ValueError("Malformed thinking output: the final answer cannot be isolated.")
    elif thinking or "<think>" in text:
        raise ValueError(
            "No completed thinking boundary was generated. Increase max_length or "
            "check the model's thinking template; no reasoning was forwarded."
        )
    # Keep H3 tags, references and user dialogue intact; remove only Qwen turn endings.
    text = re.sub(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$", "", text).strip()
    if not text:
        raise ValueError("The model produced no final answer. Increase max_length and retry.")
    return text


class FinalAnswerClip:
    """Keep native generation, but isolate reasoning during output decoding."""

    def __init__(self, clip, thinking):
        self.clip = clip
        self.thinking = thinking

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def decode(self, ids, **kwargs):
        raw = self.clip.decode(ids, skip_special_tokens=False)
        if self.thinking or "<think>" in raw or "</think>" in raw:
            return final_answer(raw, thinking=self.thinking)
        # Non-thinking outputs preserve the native decoder's exact behavior.
        return self.clip.decode(ids, **kwargs)


def collect_images(inputs):
    images, labels = [], []
    for slot in range(1, 9):
        batch = inputs.get(f"image_{slot}")
        if batch is None:
            continue
        if batch.ndim != 4 or batch.shape[-1] != 3 or min(batch.shape) < 1:
            raise ValueError(f"image_{slot} must be a non-empty IMAGE batch [B,H,W,3].")
        for frame in range(batch.shape[0]):
            images.append(batch[frame:frame + 1])
            suffix = f", batch item {frame + 1}" if batch.shape[0] > 1 else ""
            labels.append(f"<Picture {len(images)}> = image_{slot}{suffix}")
    return images, labels


def image_payloads(value):
    if isinstance(value, dict):
        if value.get("type") == "image" and "data" in value:
            yield value["data"]
        else:
            for child in value.values():
                yield from image_payloads(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from image_payloads(child)


class MultiImageClip:
    """Adapt only tokenization; keep native generation and sampling behavior."""

    def __init__(self, clip, images):
        self.clip = clip
        self.images = images

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def tokenize(self, prompt, **kwargs):
        kwargs.pop("image", None)
        tokens = self.clip.tokenize(prompt, images=self.images, **kwargs)
        attached = list(image_payloads(tokens))
        # Identity checks avoid tensor equality and catch lost, reordered or
        # duplicated references before an expensive model generation.
        if len(attached) != len(self.images) or any(
            actual is not expected for actual, expected in zip(attached, self.images)
        ):
            raise ValueError(
                f"The tokenizer did not attach all {len(self.images)} images in order "
                f"(found {len(attached)} image tokens). Use a Qwen vision CLIP with "
                "multi-image support and the default chat template. Custom templates "
                "need one vision placeholder per image."
            )
        return tokens


class TextGenerateMultiImage(TextGenerate):
    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inputs = []
        for entry in parent.inputs:
            if entry.id == "image":
                inputs.extend(
                    io.Image.Input(
                        f"image_{slot}", optional=True,
                        tooltip=(
                            "Separate visual reference. Images are sent in socket order; "
                            "every batch item is included. Different resolutions are supported."
                        ),
                    )
                    for slot in range(1, 9)
                )
            else:
                inputs.append(entry)
        return io.Schema(
            node_id="TextGenerateMultiImage",
            display_name="Vision Prompt Composer",
            category="text",
            description=(
                "Generate one text from up to eight image inputs using a Qwen vision CLIP. "
                "References are numbered <Picture N> in connected-input order. "
                "All batch items are included without resizing or creating a collage."
            ),
            search_aliases=["LLM", "Qwen", "MiniMax", "multiple images", "Generate Text", "Multi Image"],
            inputs=inputs,
            outputs=parent.outputs,
        )

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, thinking=False,
                use_default_template=True, video=None, audio=None, **kwargs):
        images, labels = collect_images(kwargs)
        if images:
            prompt += (
                "\n\nVisual reference index (images supplied in this exact order):\n"
                + "\n".join(labels)
                + "\nInspect every supplied image when writing the requested prompt. "
                "Keep the references distinct and use these <Picture N> labels consistently. "
                "Reference roles come from the user's request; numbering alone does not "
                "assign a first-frame, last-frame, identity or style role."
            )
            clip = MultiImageClip(clip, images)
        return super().execute(
            FinalAnswerClip(clip, thinking), prompt, max_length, sampling_mode, thinking=thinking,
            use_default_template=use_default_template, video=video, audio=audio,
        )


class MultiImageTextExtension(ComfyExtension):
    async def get_node_list(self):
        return [TextGenerateMultiImage]


async def comfy_entrypoint():
    return MultiImageTextExtension()

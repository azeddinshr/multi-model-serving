import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODELS = {
    "tiny_sentiment": {
        "hf_name": "sshleifer/tiny-distilbert-base-cased-finetuned-sst-2-english",
        "output_dir": "model_repository/tiny_sentiment/1/model.onnx",
    },
    "tiny_spam": {
        "hf_name": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
        "output_dir": "model_repository/tiny_spam/1/model.onnx",
    },
}


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


for model_name, config in MODELS.items():
    print(f"\nExporting {model_name}")
    print(f"HF model: {config['hf_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["hf_name"]
    )
    model.eval()

    encoded = tokenizer(
        "This is a test sentence.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    wrapped = Wrapper(model)

    os.makedirs(os.path.dirname(config["output_dir"]), exist_ok=True)

    torch.onnx.export(
        wrapped,
        (
            encoded["input_ids"],
            encoded["attention_mask"],
        ),
        config["output_dir"],
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=17,
    )

    print(f"Saved: {config['output_dir']}")

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftType  # type: ignore


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, lora_modules=None, linear_probe=False, lora_rank=8, lora_alpha=8, lora_dropout=0.0, **kwargs):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        if lora_modules is not None:
            peft_config = LoraConfig(
                peft_type=PeftType.LORA,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=lora_modules,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            lm = get_peft_model(lm, peft_config)
            self.transformer = lm.base_model.base_model # PeftModel -> LoraModel -> PreTrainedModel
        else:
            self.transformer = lm.base_model  # CausalLM -> PreTrainedModel
        lm_head = getattr(lm, "lm_head", getattr(lm, "embed_out", None))
        assert isinstance(lm_head, torch.nn.Linear)
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        assert isinstance(hidden_size, int)
        self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
            lm_head.weight.dtype
        )
        torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [
                transformer_outputs[0][i, input_lens[i] - 1, :]
                for i in range(len(input_lens))
            ]
        )
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits

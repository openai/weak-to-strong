from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftType  # type: ignore
from typing import Optional


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(
        self,
        name,
        lora_modules=None,
        use_lm_head=False,
        linear_probe=False,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_lm_head = use_lm_head
        self.lora_modules = lora_modules
        self.lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        if lora_modules is not None:
            print(f"Using LoraModel on modules {lora_modules}")
            peft_config = LoraConfig(
                peft_type=PeftType.LORA,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=lora_modules,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.lm = get_peft_model(self.lm, peft_config)

        lm_head = getattr(self.lm, "lm_head", getattr(self.lm, "embed_out", None))
        assert isinstance(lm_head, torch.nn.Linear)
        if use_lm_head:
            print("Using LM head instead of learned head because choices are provided")
            self.score = None
        else:
            hidden_size = getattr(
                config, "n_embd", getattr(config, "hidden_size", None)
            )
            assert isinstance(hidden_size, int)
            self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
                lm_head.weight.dtype
            )
            torch.nn.init.normal_(self.score.weight, std=0.0)
        self.linear_probe = linear_probe

    @property
    def transformer(self):
        if self.lora_modules is not None:
            return (
                self.lm.base_model.base_model
            )  # PeftModel -> LoraModel -> PreTrainedModel
        return self.lm.base_model  # CausalLM -> PreTrainedModel

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)
    
    def save_torch(self, path, optimizer=None, scheduler=None):
        save_dict = self.state_dict()
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        torch.save(save_dict, path)

    def gradient_checkpointing_enable(self):
        model = self.transformer if self.score is not None else self.lm
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor,
        choice_input_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)

        if self.score is None:  # use LM head
            assert choice_input_ids is not None
            all_logits = self.lm(input_ids).logits
            logits_at_last = [
                all_logits[i, input_lens[i] - 1, choice_input_ids[i]]
                for i in range(len(input_lens))
            ]  # [batch_size, num_choices]
            logits = torch.stack(logits_at_last)
        else:  # use learned head
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

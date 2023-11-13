import torch
import torch.nn as nn
from transformers import T5Model

class t5ForStoryGeneration(nn.Module):
    def __init__(self, checkpoint):
        super(t5ForStoryGeneration, self).__init__()
        self.model = T5Model.from_pretrained(checkpoint)
        # self.extra_linear = nn.Linear(self.model.config.hidden_size, extra_dim)

        # Freeze the parameters of the T5 model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=labels)
        last_hidden_states = outputs.last_hidden_state
        custom_output = self.extra_linear(last_hidden_states)

         # If labels are provided, calculate loss (otherwise, skip this part)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(custom_output.view(-1, custom_output.size(-1)), labels.view(-1))

        return custom_output, loss
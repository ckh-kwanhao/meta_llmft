
        logits = self.lm_head(outputs[0])

        # In the classification setting we only care about the last prediction
        # Get the position of the last non-padding token
        sequence_lengths = torch.ne(
            input_ids, self.config.pad_token_id).sum(-1) - 1
        logits = logits[torch.arange(
            input_ids.shape[0], device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            logits = logits.contiguous()
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            labels = labels.contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        # we overwrite this function to support untying the input and output embeddings
        
        if not self.config.untie_embeddings:
            # default implementation from hf transformers
            if self.config.torchscript:
                output_embeddings.weight = nn.Parameter(
                    input_embeddings.weight.clone())
            else:
                output_embeddings.weight = input_embeddings.weight

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0] -
                        output_embeddings.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings
        else:
            # do nothing
            print("**** Untying input and output embeddings ****")

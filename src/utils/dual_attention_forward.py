from diffusers.models.attention import BasicTransformerBlock

def dual_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    timestep=None,
    cross_attention_kwargs=None,
    class_labels=None,
):

    # ===== Self-Attention =====
    norm_hidden_states = self.norm1(hidden_states)

    sa_out = self.attn1(
        norm_hidden_states,
        attention_mask=attention_mask,
        encoder_hidden_states=None,
    )

    # ===== Cross-Attention =====
    norm_hidden_states_cross = self.norm2(hidden_states)

    ca_out = self.attn2(
        norm_hidden_states_cross,
        attention_mask=encoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
    )

    # ===== Fusion =====
    alpha = 0.5
    attn_out = alpha * sa_out + (1 - alpha) * ca_out

    hidden_states = hidden_states + attn_out

    # ===== Feed Forward =====
    norm_hidden_states = self.norm3(hidden_states)
    ff_out = self.ff(norm_hidden_states)

    hidden_states = hidden_states + ff_out

    return hidden_states
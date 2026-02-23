import torch
from diffusers.models.attention import CrossAttention

class MyCrossAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,             # 當前這一層的 Attention 模組（UNet 裡的一個 attention layer）
        hidden_states,                    # Query 來源 (latent feature)
        encoder_hidden_states=None,       # Key / Value 來源 (文字 embedding 或 self)
        attention_mask=None
    ):
        # hidden_states: (B, N, C)
        # B = batch size
        # N = token 數 (例如 64x64 latent flatten 後的 token)
        # C = channel dimension

        batch_size, sequence_length, _ = hidden_states.shape

        # 準備 attention mask（通常 diffusion 不太用到）
        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            sequence_length
        )

        # ===============================
        # 1️⃣ 產生 Query
        # ===============================
        # Q = W_q * hidden_states
        query = attn.to_q(hidden_states)

        # ===============================
        # 2️⃣ 決定是 cross 還是 self attention
        # ===============================
        # 如果 encoder_hidden_states 是 None
        # → 使用 hidden_states 本身
        # → 就變成 self-attention
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        # ===============================
        # 3️⃣ 產生 Key / Value
        # ===============================
        # K = W_k * encoder_hidden_states
        # V = W_v * encoder_hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # ===============================
        # 4️⃣ Multi-Head reshape
        # ===============================
        # 將 (B, N, C) 轉成 (B * heads, N, head_dim)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # ===============================
        # 5️⃣ 計算 Attention 分數
        # ===============================
        # attention_probs = softmax(QK^T / sqrt(d))
        attention_probs = attn.get_attention_scores(
            query,
            key,
            attention_mask
        )

        # ⭐ 這行是 pix2pix-zero 新增的
        # 把 attention map 存起來
        # 方便之後做 loss / 控制 / 對齊
        attn.attn_probs = attention_probs

        # ===============================
        # 6️⃣ Attention 加權
        # ===============================
        # output = Attention * V
        hidden_states = torch.bmm(attention_probs, value)

        # 把 multi-head 維度合併回來
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # ===============================
        # 7️⃣ 最後線性投影
        # ===============================
        hidden_states = attn.to_out[0](hidden_states)

        # dropout（推論時通常沒影響）
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MyDualAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        weight_self = 0.5
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        # ==========================================
        # 1️⃣ Query (共用)
        # ==========================================
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        # ==========================================
        # 2️⃣ Self-Attention branch
        #    K,V 來自 hidden_states
        # ==========================================
        key_self = attn.to_k(hidden_states)
        value_self = attn.to_v(hidden_states)

        key_self = attn.head_to_batch_dim(key_self)
        value_self = attn.head_to_batch_dim(value_self)

        attn_probs_self = attn.get_attention_scores(query, key_self, attention_mask)
        hidden_self = torch.bmm(attn_probs_self, value_self)

        # ==========================================
        # 3️⃣ Cross-Attention branch
        #    K,V 來自 encoder_hidden_states (text)
        # ==========================================
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key_cross = attn.to_k(encoder_hidden_states)
        value_cross = attn.to_v(encoder_hidden_states)

        key_cross = attn.head_to_batch_dim(key_cross)
        value_cross = attn.head_to_batch_dim(value_cross)

        attn_probs_cross = attn.get_attention_scores(query, key_cross, attention_mask)
        hidden_cross = torch.bmm(attn_probs_cross, value_cross)

        # ==========================================
        # 4️⃣ 加權融合 (alpha / (1-alpha))
        # ==========================================
        hidden_states = weight_self * hidden_self + (1 - weight_self) * hidden_cross

        # 還原 head 維度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # ==========================================
        # 5️⃣ output projection
        # ==========================================
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
# attn1: self-attention
# attn2: cross-attention
def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
            
    # replace the fwd function
    # 替換cross-attention的forward processor
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            #module.set_processor(MyCrossAttnProcessor())
            module.set_processor(MyDualAttnProcessor())
    return unet


def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        # requires_grad代表哪些參數可以被訓練更新
        # 改成self-attention跟cross-attention都可以訓練
        if 'attn1' in name or 'attn2' in name:  
            params.requires_grad = True
        else:
            params.requires_grad = False

    # 替換self-attention跟cross-attention的forward processor
    # self-attention跟cross-attention都是在CrossAttention類別下，分別在於dim的不同
    # module.cross_attention_dim == None: self-attention
    # module.cross_attention_dim == 768: cross-attention
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_processor(MyCrossAttnProcessor())
    return unet
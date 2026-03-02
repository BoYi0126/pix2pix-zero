import pdb, sys

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet
import json


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EditingPipeline(BasePipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,    # latent gradient step size
        edit_dir=None,  # 語意方向
        x_in=None,  # 輸入latent (通常由inversion得到)
        only_sample=False, # only perform sampling, and no editing # 只做reconstruction，不做editing

    ):

        x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 0. modify the unet to be useful :D
        # Step0. 修改unet，在unet的corss-attention模組內部加入module.attn_probs，為了後面可以計算loss
        self.unet = prep_unet(self.unet)
        
        # 1. setup all caching objects
        # 是 timestep → layer → attention map 的對應表。
        d_ref_t2attn = {} # reference cross attention maps
        d_ref_t1attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        # 因為diffusion在latent space運作，latent = H/8 x W/8
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device # 設定device
        do_classifier_free_guidance = guidance_scale > 1.0 # 若guidance_scale > 1.0，則為true，後續會根據這個旗標做事情 (這是CFG的開關 classifier-free guidance)
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device) # 把x_in移到正確的device，把x_in轉成和unet相同的dtype (如果x_in是float32，unet是float16就會有問題，所以要執行這行)
        
        # 3. Encode input prompt = 2x77x1024
        # 若使用classifier-free guidance: embedding 會變成 [negative_prompt, positive_prompt]
        # 把文字prompt轉成可供unet cross-attention使用的embedding tensor
        # text → tokenizer → text_encoder → embedding → UNet cross-attention (Stable Diffusion 並不是把文字直接丟進 UNet。)
        prompt_embeds = self._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)

        # 取得對應的 noise schedule
        sigmas = None
        alphas_cumprod = None
        if hasattr(self.scheduler, "sigmas"):
            sigmas = self.scheduler.sigmas
        else:
            # fallback: 用 alphas_cumprod 計算 noise 強度
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
    
        # 4. Prepare timesteps
        # 建立diffusion timesteps，這決定x_t -> x_0的反向去噪順序
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        
        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        # 第一輪Denoising，不算梯度，只收集attention
        # 先跑一次 reference image，儲存每個 timestep 的 cross-attention map
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with torch.no_grad(): # 代表在這個區塊內「不建立計算圖、不追蹤梯度」，也就是只做純前向推論，不會進行反向傳播。
            with self.progress_bar(total=num_inference_steps) as progress_bar: # 建立一個進度條物件，在diffusion迴圈中顯示目前的執行進度
                # 在迭代的同時取得索引i以及元素t，實際的timesteps可能是: tensor([999, 979, 959, ..., 19, 0])
                # 所以第一個元素是 i=0, t=999，第二個是i=1, t=979, ...
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    
                    # 為CFG準備batch，如果使用classifier-free guidance (CFG)，unet需同時計算: 1.無條件 ε_uncond 2.有條件 ε_text，所以把batch複製成兩份
                    # tensor的格式: (batch_size, channels, height, width)
                    # 假設原本是 (1, 4, 64, 64)，會變成(2, 4, 64, 64)，所以是batch_size變成兩倍
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    
                    # 把x_t調整成符合該scheduler理論假設的輸入格式，確保訓練跟推理時的數學公式一致
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    # 預測噪音
                    noise_pred = self.unet(latent_model_input,
                                           t,
                                           encoder_hidden_states=prompt_embeds,
                                           cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {} # 建立timestep的容器，t.item()是轉成整數，d_ref_t2attn是字典 (等於是為了目前的timestep建立一個空的attention儲存空間)
                    d_ref_t1attn[t.item()] = {} # 建立timestep的容器，t.item()是轉成整數，d_ref_t2attn是字典 (等於是為了目前的timestep建立一個空的attention儲存空間)
                    '''
                    結構如下
                    d_ref_t2attn = {
                        t0: { ... },
                        t1: { ... },
                    }
                    '''
                    # 掃描整個unet，會遍歷down blocks, mid blocks, up blocks
                    # name: 層的路徑名稱
                    # module: 實際的module
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        # 儲存attention (A_ref)
                        if module_name == "CrossAttention" and 'attn2' in name:
                            # 取得attention map，attn_probs通常是 softmax(QK^T / sqrt(d))
                            # num_channel: attention head數或是batch*heads
                            # s*s: 例如64*64，視該層解析度
                            # 77: CLIP text token數
                            # 這代表每個 spatial 位置 對 每個文字 token 的注意力權重
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            
                            # detach(): 把它從 computational graph 拔掉 → 不會參與梯度回傳，代表這只是記錄 reference attention
                            # cpu(): 把 tensor 移到 CPU → 節省 GPU 記憶體
                            # 存到dict，最後結構會像這樣
                            '''
                            d_ref_t2attn = {
                                t: {
                                    "down_blocks.0.attn2": tensor(...),
                                    "mid_block.attn2": tensor(...),
                                    ...
                                }
                            }
                            '''
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()
                            
                        if module_name == "CrossAttention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t1attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    # 做CFG (放大條件方向)
                    if do_classifier_free_guidance:
                        # 前面有提到batch的大小會變成兩倍，這邊就是沿著batch的維度拆解成兩份
                        # noise_pred_uncond → 無條件預測
                        # noise_pred_text → 有文字條件預測
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        # ϵ_guided​=ϵ_uncond​+w(ϵ_text​−ϵ_uncond​)，w = guidance_scale
                        # 如果 guidance_scale = 0，則 noise_pred = noise_pred_uncond
                        # 如果 guidance_scale = 1，則 noise_pred = noise_pred_text
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # Scheduler 更新 latent: 用預測的 noise從 x_t 計算出 x_{t-1}，這行其實就是latents = x_{t-1}
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    # 更新進度條
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        # latents.detach(): 把 tensor 從 computational graph 中分離。不再追蹤梯度，純粹當成數值
        # self.decode_latents(): x_reconstruction = VAE_decoder(z), x_reconstruction: decoder後的圖片, z: latent內的特徵
        # 這行等於是把 (1, 4, 64, 64)還原成(1, 3, 512, 512)，經過訓練好的 VAE decoder，把壓縮的表示還原成 RGB 圖
        # decode_latents()回傳的通常是numpy array，[-1, 1] 或 [0, 1]
        # numpy_to_pil(): 1. 轉成 uint8, 2.scale 到 [0,255], 3: 轉成 PIL Image 物件 (可以直接.show()或是.save())
        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))

        if only_sample:
            return image_rec

        # 取得prompt
        prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        # 通常prompt_embeds_edit會有三個維度 (2, 77, 768), dim1: 2, dim2: 77, dim3: 768
        # 假設edit_dir是(1, 77, 768)，若使用整數索引: prompt_embeds_edit[1]的話會降維，會得到(77, 768)，兩個維度不同無法相加
        # 所以要用slice索引: [1:2] 取index 1~index 2但不包含 index 2，會得到(1, 77, 768)
        prompt_embeds_edit[1:2] += edit_dir
        
        # 取得原本的latents，準備做另一個denoising迴圈
        latents = latents_init
        
        # Second denoising loop for editing the text prompt
        sigma_record = [] # 紀錄每個timestep的noise強度，方便後面分析
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # 取得Tau，這是用來區分高噪聲和低噪聲的閾值，在高噪聲階段使用Loss優化，在低噪聲階段正常去噪
        noise_threshold = 0.5 # 這個值是經過實驗調整的，代表在noise強度超過這個值的階段，我們認為是高噪聲階段，適合用Loss優化
        tau_method = "median" # 這個值是經過實驗調整的，代表用來計算tau的方法，"value_percent"代表用噪聲強度，"median"代表用noise schedule的中位數
        if tau_method == "value_percent":
            if hasattr(self.scheduler, "sigmas"):
                tau = sigmas.max() * noise_threshold   # 高噪聲前半段
            else:
                tau = torch.sqrt(1 - alphas_cumprod.min()) * noise_threshold
        elif tau_method == "median":
            if hasattr(self.scheduler, "sigmas"):
                tau = torch.quantile(sigmas, 0.5)
            else:
                sigma_all = torch.sqrt(1 - alphas_cumprod)
                tau = torch.quantile(sigma_all, 0.5)
                
        with self.progress_bar(total=num_inference_steps) as progress_bar:# 建立一個進度條物件，在diffusion迴圈中顯示目前的執行進度
            # 在迭代的同時取得索引i以及元素t，實際的timesteps可能是: tensor([999, 979, 959, ..., 19, 0])
            # 所以第一個元素是 i=0, t=999，第二個是i=1, t=979, ...
            for i, t in enumerate(timesteps):
                

                # 取得目前 timestep 的 noise 強度
                if hasattr(self.scheduler, "sigmas"):
                    sigma_t = sigmas[i]
                else:
                    alpha_t = alphas_cumprod[t]
                    sigma_t = torch.sqrt(1 - alpha_t)
                
                # 儲存sigma的資訊 後面方便比對
                sigma_record.append({
                    "step_index": i,
                    "timestep": int(t.item()),
                    "sigma": float(sigma_t.detach().cpu().item())
                })
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents # 參考第一輪denoising的說明，code都一樣
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)    # 參考第一輪denoising的說明，code都一樣
                    
                # 前50%照樣用Loss做優化
                #if i < int(len(timesteps) * 0.5):
                if sigma_t > tau:
                    # 切斷舊的 computation graph，讓 x_in 成為全新的葉節點 (leaf tensor)，然後只對 x_in 做優化。
                    # clone() 只做：複製 tensor 的數值，但它 保留 gradient graph 連結。
                    # detach() 的意思是：把 tensor 從原本的 graph 中拔掉
                    # 這邊雖然還是不太懂為什麼graph會有連結，不過就先這樣
                    x_in = latent_model_input.detach().clone()
                    
                    # 讓latent可以被優化
                    x_in.requires_grad = True
                    
                    # 建立 optimizer(優化器)，代表每個 timestep 會對 latent 做 SGD 更新
                    # SGD-準確率梯度下降法 (stochastic gradient decent)
                    opt = torch.optim.SGD([x_in], lr=guidance_amount)

                    # predict the noise residual
                    noise_pred = self.unet(x_in,
                                        t,
                                        encoder_hidden_states=prompt_embeds_edit.detach(),
                                        cross_attention_kwargs=cross_attention_kwargs,).sample
                    
                    # 讓這輪的cross-attention接近第一輪的cross-attention
                    loss = 0.0
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "CrossAttention" and 'attn2' in name:
                            curr = module.attn_probs # size is num_channel,s*s,77
                            ref = d_ref_t2attn[t.item()][name].detach().to(device)  # 取得第一輪的attention map
                            loss += ((curr-ref)**2).sum((1,2)).mean(0)
                    loss.backward(retain_graph=False)
                    opt.step()

                    # recompute the noise
                    with torch.no_grad():
                        noise_pred = self.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                    
                    latents = x_in.detach().chunk(2)[0]
                else:   # 剩餘的部分這邊想改成使用self-attention
                    self_attenation_enable = 0
                    if self_attenation_enable == 0:
                        with torch.no_grad():  
                            # 預測噪音
                            noise_pred = self.unet(latent_model_input,
                                                t,
                                                encoder_hidden_states=prompt_embeds_edit.detach(),
                                                cross_attention_kwargs=cross_attention_kwargs,).sample
                    else:
                        x_in = latent_model_input.detach().clone()
                        x_in.requires_grad = True
                        opt = torch.optim.SGD([x_in], lr=guidance_amount)

                        noise_pred = self.unet(
                            x_in,
                            t,
                            encoder_hidden_states=prompt_embeds_edit.detach(),
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        loss = 0.0
                        for name, module in self.unet.named_modules():
                            module_name = type(module).__name__
                            if module_name == "CrossAttention" and 'attn1' in name:
                                curr = module.attn_probs
                                ref = d_ref_t1attn[t.item()][name].detach().to(device)
                                loss += ((curr - ref) ** 2).sum((1, 2)).mean(0)

                        loss.backward()
                        opt.step()

                        with torch.no_grad():
                            noise_pred = self.unet(
                                x_in.detach(),
                                t,
                                encoder_hidden_states=prompt_embeds_edit,
                                cross_attention_kwargs=cross_attention_kwargs,
                            ).sample

                        latents = x_in.detach().chunk(2)[0]
                        
                            
                # perform guidance
                # 跟第一個迴圈code一樣
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        # 8. Post-processing
        image = self.decode_latents(latents.detach())

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image_edit = self.numpy_to_pil(image)

        # 將sigma儲存成檔案
        with open("sigma_log.json", "w") as f:
            json.dump(sigma_record, f, indent=4)


        return image_rec, image_edit

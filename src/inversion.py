# ===== 基本套件 =====
import os, pdb                      # os: 檔案與路徑操作；pdb: 除錯工具
from glob import glob               # 讀取資料夾中符合條件的檔案
import argparse                     # 解析命令列參數
import numpy as np
import torch
import requests
from PIL import Image               # 影像讀取與處理

# ===== BLIP 模型 (自動產生 caption) =====
from lavis.models import load_model_and_preprocess

# ===== 自訂 DDIM inversion pipeline =====
from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler


# ===== 裝置設定 =====
# 若有 GPU 則使用 GPU，否則使用 CPU
if torch.cuda.is_available():
    device = "cuda"
    print("[inversion.py] Device: CUDA")
    print("[inversion.py] GPU Name:", torch.cuda.get_device_name(0))
    print("[inversion.py] CUDA Version:", torch.version.cuda)
else:
    device = "cpu"
    print("[inversion.py] CPU")


if __name__=="__main__":

    # ===== 1️⃣ 解析命令列參數 =====
    parser = argparse.ArgumentParser()
    
    # 輸入影像 (可為單張圖片或資料夾)
    parser.add_argument('--input_image', type=str, default='assets/test_images/cat_a.png')
    
    # 輸出資料夾
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    
    # DDIM inversion 的步數
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    
    # Stable Diffusion 權重路徑
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    
    # 是否使用 float16 (節省顯存)
    parser.add_argument('--use_float_16', action='store_true')
    
    args = parser.parse_args()


    # ===== 2️⃣ 建立輸出資料夾 =====
    # inversion: 儲存 latent inversion 結果
    # prompt: 儲存自動生成的文字描述
    os.makedirs(os.path.join(args.results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)


    # ===== 3️⃣ 設定 tensor 精度 =====
    # float16 可減少顯存消耗，但可能降低數值穩定性
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    # ===== 4️⃣ 載入 BLIP caption 模型 =====
    # BLIP: Bootstrapping Language-Image Pretraining
    # 用來自動將輸入影像轉為文字 prompt
    model_blip, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",          # caption 任務
        model_type="base_coco",       # 使用 COCO 訓練版本
        is_eval=True,                 # inference 模式
        device=torch.device(device)
    )


    # ===== 5️⃣ 建立 DDIM Inversion pipeline =====
    # 從 Stable Diffusion 權重載入模型, pipe的物件型別是DDIMInversion(雖然是DiffusionPipeline的屬性，但因為是classMethod所以會回傳呼叫的那個類別)
    # 建立一個DDIM Inversion物件，並且移到GPU
    pipe = DDIMInversion.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype
    ).to(device)

    # 替換 scheduler 為「反向 DDIM scheduler」
    # 用於執行 deterministic inversion
    # scheduler 決定每一步 diffusion 的時間步與更新公式；在 inversion 時必須改成「反向更新公式」，所以要替換成 DDIMInverseScheduler。
    # UNet 負責預測，Scheduler 負責算下一步 latent
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)


    # ===== 6️⃣ 判斷輸入是單張圖片還是資料夾 =====
    if os.path.isdir(args.input_image):
        # 若是資料夾，收集所有 .png 圖片
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        # 若是單張圖片，轉成 list
        l_img_paths = [args.input_image]


    # ===== 7️⃣ 對每張圖片做 inversion =====
    for img_path in l_img_paths:

        # 取得檔名（不含副檔名）
        bname = os.path.basename(img_path).split(".")[0]

        # 讀取圖片並 resize 至 512x512
        # Stable Diffusion v1-4 預設輸入大小為 512x512
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)


        # ===== 7-1️⃣ 使用 BLIP 產生 caption =====
        # 將 PIL image 轉為 tensor 並做標準化
        _image = vis_processors["eval"](img).unsqueeze(0).to(device)

        # 生成文字描述 (prompt)
        prompt_str = model_blip.generate({"image": _image})[0]


        # ===== 7-2️⃣ 執行 DDIM inversion =====
        # 將影像反推回 diffusion latent space
        # x_inv: inversion 過程中的 latent
        # x_inv_image: inversion 對應影像
        # x_dec_img: 重建後影像
        # 這邊是呼叫pipe.call
        x_inv, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,                  # 不使用 classifier-free guidance
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )


        # ===== 7-3️⃣ 儲存 inversion latent =====
        torch.save(
            x_inv[0], 
            os.path.join(args.results_folder, f"inversion/{bname}.pt")
        )


        # ===== 7-4️⃣ 儲存自動生成的 prompt =====
        with open(
            os.path.join(args.results_folder, f"prompt/{bname}.txt"), 
            "w"
        ) as f:
            f.write(prompt_str)
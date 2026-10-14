from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

app = FastAPI(
    title="SoloHash Image Feature Extractor",
    description="Extract Sobel image feature vectors from uploaded picture",
    version="1.0.0"
)

# 設備：雲端可以切 CPU / 雲GPU，唔需要你屋企顯卡
device = torch.device("cpu")

# 核心Sobel特徵提取（完全重用你SoloHash嘅演算法）
def extract_image_feature(img: Image.Image):
    trans = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    img_tensor = trans(img.convert("L")).unsqueeze(0).to(device)

    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    feat_x = torch.nn.functional.conv2d(img_tensor, sobel_x, padding=1)
    feat_y = torch.nn.functional.conv2d(img_tensor, sobel_y, padding=1)
    feature_map = torch.sqrt(feat_x**2 + feat_y**2)

    feat_mean = torch.mean(feature_map, dim=(2,3))
    feat_std = torch.std(feature_map, dim=(2,3))
    feature_vector = torch.cat([feat_mean, feat_std], dim=1)

    return feature_vector.squeeze().detach().cpu().numpy().tolist()

@app.post("/extract-feature")
async def extract_feature(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        feature_vec = extract_image_feature(image)
        return JSONResponse(content={
            "success": True,
            "feature_vector": feature_vec
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)

@app.get("/")
async def root():
    return {"msg":"SoloHash API is running. POST to /extract-feature to process image."}

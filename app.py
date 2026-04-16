import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

# --- 1. Model Architecture (U-Net) ---
class UNetDenoiser(nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.pool(e1)
        e2 = self.relu(self.enc2(e2))
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.relu(self.dec1(d1))
        out = self.sigmoid(self.dec2(d1))
        return out

# --- 2. Load the Saved Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetDenoiser().to(device)
# පල්ලෙහා තියෙන ෆයිල් එක උඹේ ලැප් එකේ තියෙන path එකට හරියටම දෙන්න
model.load_state_dict(torch.load('tem_denoiser_unet.pth', map_location=device))
model.eval()

# --- 3. Streamlit UI Design ---
st.title("🔬 AI-Powered TEM Image Denoiser")
st.subheader("Developed by Eshan Alahakoon (USJP)")
st.write("Upload a noisy TEM image to clean it using Deep Learning (U-Net).")

uploaded_file = st.file_uploader("Choose a TEM image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Original පින්තූරය load කිරීම
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    
    # පින්තූරයේ Original size එක මතක තබා ගැනීම (Width, Height)
    original_size = image.size 
    
    st.image(image, caption='Uploaded Image (Noisy)', use_column_width=True)
    
    if st.button('Clean Image'):
        # 2. AI එකට ලේසි වෙන්න 256x256 ට resize කිරීම
        img_input = cv2.resize(img_array, (256, 256))
        img_tensor = torch.from_numpy(img_input.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        # 3. AI output එක ගන්නවා
        denoised_img = output.squeeze().cpu().numpy()
        
        # 4. වැදගත්ම කොටස: ආපහු Original Resolution එකට Resize කිරීම
        # මෙතනදී original_size පාවිච්චි කරන නිසා පින්තූරය ඇද වෙන්නේ නැහැ
        denoised_resized = cv2.resize(denoised_img, original_size)
        
        denoised_final = (denoised_resized * 255).astype(np.uint8)
        
        # Result එක පෙන්වීම
        st.success("Denoising Complete!")
        st.image(denoised_final, caption='AI Denoised Result (Original Resolution)', use_column_width=True)
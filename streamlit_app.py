import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# 1. 網頁標題與設定
st.set_page_config(page_title="互動式 AI 影像處理與人臉偵測系統", layout="wide")
st.title("🎨 互動式 AI 影像處理與人臉偵測系統")
st.write("請在左側側邊欄上傳圖片並調整參數！")

# 2. 載入 OpenCV 內建的人臉偵測分類器 (不使用快取以避免 Streamlit 報錯)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. 側邊欄控制面版
st.sidebar.header("控制面版")
uploaded_file = st.sidebar.file_uploader("上傳一張圖片", type=["jpg", "jpeg", "png"])

filter_type = st.sidebar.radio(
    "選擇處理功能",
    [
        "原圖", 
        "人臉自動打馬賽克 (Mosaic)", 
        "人臉標記框 (Face Detection)",
        "高斯模糊", 
        "AI 邊緣偵測 (Canny)", 
        "經典黑白", 
        "復古暖色調"
    ]
)

# 根據選取的模式動態顯示對應參數
mosaic_size = 15
blur_amount = 5
brightness = 1.0

if filter_type == "人臉自動打馬賽克 (Mosaic)":
    mosaic_size = st.sidebar.slider("馬賽克顆粒大小", min_value=5, max_value=50, value=20, step=5)
elif filter_type == "高斯模糊":
    blur_amount = st.sidebar.slider("模糊程度", min_value=1, max_value=20, value=5)

brightness = st.sidebar.slider("亮度調整", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# 4. 影像處理核心邏輯
if uploaded_file is not None:
    # 讀取圖片並轉為 numpy 陣列
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    
    # 調整亮度
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    img = np.array(enhancer.enhance(brightness))
    
    # 功能切換判斷
    if filter_type == "人臉自動打馬賽克 (Mosaic)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 針對每一張偵測到的臉進行馬賽克處理
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            # 縮小再放大產生馬賽克效果
            mh = max(1, h // mosaic_size)
            mw = max(1, w // mosaic_size)
            small_face = cv2.resize(face_roi, (mw, mh), interpolation=cv2.INTER_LINEAR)
            mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y:y+h, x:x+w] = mosaic_face
            
        st.sidebar.success(f"偵測到 {len(faces)} 張人臉！")

    elif filter_type == "人臉標記框 (Face Detection)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 畫出人臉綠色框線
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        st.sidebar.success(f"偵測到 {len(faces)} 張人臉！")

    elif filter_type == "高斯模糊":
        ksize = int(blur_amount) * 2 + 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif filter_type == "AI 邊緣偵測 (Canny)":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif filter_type == "經典黑白":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif filter_type == "復古暖色調":
        r, g, b = cv2.split(img)
        r = cv2.add(r, 30)
        b = cv2.subtract(b, 30)
        img = cv2.merge((r, g, b))
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 左右排版展示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原圖")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("處理後結果")
        st.image(img, use_column_width=True)
        
        # 新增下載處理後圖片的按鈕
        result_pil = Image.fromarray(img)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ 下載處理後的圖片",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )
else:
    st.info("👈 請先從左側側邊欄上傳包含人臉的照片試試看！")

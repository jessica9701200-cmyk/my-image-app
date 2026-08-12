import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 1. 網頁標題與說明
st.set_page_config(page_title="互動式 AI 影像處理系統", layout="wide")
st.title("🎨 互動式 AI 影像處理系統")
st.write("請在左側側邊欄上傳圖片並調整參數！")

# 2. 側邊欄控制控制板
st.sidebar.header("控制面版")
uploaded_file = st.sidebar.file_uploader("上傳一張圖片", type=["jpg", "jpeg", "png"])

filter_type = st.sidebar.radio(
    "選擇濾鏡效果",
    ["原圖", "高斯模糊", "AI 邊緣偵測 (Canny)", "經典黑白", "復古暖色調"]
)

blur_amount = st.sidebar.slider("模糊程度", min_value=1, max_value=20, value=5)
brightness = st.sidebar.slider("亮度調整", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# 3. 影像處理邏輯與畫面渲染
if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # 調整亮度
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    img = np.array(enhancer.enhance(brightness))
    
    # 根據選取套用濾鏡
    if filter_type == "高斯模糊":
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

    # 左右排版顯示原始與處理後的圖片
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原圖")
       st.image(image, use_container_width=True
    with col2:
        st.subheader("處理後結果")
       st.image(img, use_container_width=True)
else:
    st.info("👈 請先從左側側邊欄上傳圖片以開始操作！")

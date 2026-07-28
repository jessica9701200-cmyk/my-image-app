import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import mediapipe as mp

# 1. 網頁標題與設定
st.set_page_config(page_title="互動式 AI 影像處理與人臉偵測系統", layout="wide")
st.title("🎨 互動式 AI 影像處理與人臉偵測系統")
st.write("請在左側側邊欄上傳圖片並調整參數！")

# 2. 初始化 MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

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
    if filter_type in ["人臉自動打馬賽克 (Mosaic)", "人臉標記框 (Face Detection)"]:
        h, w, _ = img.shape
        face_count = 0
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img)
            
            if results.detections:
                face_count = len(results.detections)
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    xmin = int(bboxC.xmin * w)
                    ymin = int(bboxC.ymin * h)
                    box_width = int(bboxC.width * w)
                    box_height = int(bboxC.height * h)
                    
                    # 邊界限制防止溢出
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmin + box_width)
                    ymax = min(h, ymin + box_height)
                    
                    if filter_type == "人臉自動打馬賽克 (Mosaic)":
                        face_roi = img[ymin:ymax, xmin:xmax]
                        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                            face_pil = Image.fromarray(face_roi)
                            small_w = max(1, face_pil.width // mosaic_size)
                            small_h = max(1, face_pil.height // mosaic_size)
                            small_face = face_pil.resize((small_w, small_h), Image.Resampling.NEAREST)
                            mosaic_face = small_face.resize((face_pil.width, face_pil.height), Image.Resampling.NEAREST)
                            img[ymin:ymax, xmin:xmax] = np.array(mosaic_face)
                            
                    elif filter_type == "人臉標記框 (Face Detection)":
                        thickness = max(2, int(min(h, w) * 0.005))
                        img[ymin:ymin+thickness, xmin:xmax] = [0, 255, 0]
                        img[ymax-thickness:ymax, xmin:xmax] = [0, 255, 0]
                        img[ymin:ymax, xmin:xmin+thickness] = [0, 255, 0]
                        img[ymin:ymax, xmax-thickness:xmax] = [0, 255, 0]

        st.sidebar.success(f"偵測到 {face_count} 張人臉！")

    elif filter_type == "高斯模糊":
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=blur_amount)))

    elif filter_type == "經典黑白":
        pil_img = Image.fromarray(img).convert("L")
        img = np.array(pil_img.convert("RGB"))

    elif filter_type == "復古暖色調":
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        r = np.clip(r.astype(int) + 30, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(int) - 30, 0, 255).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)

    # 左右排版展示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原圖")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("處理後結果")
        st.image(img, use_column_width=True)
        
        # 下載圖片按鈕
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

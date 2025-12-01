import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

st.set_page_config(
    page_title="Minecraft Animal Detector",
    page_icon="🟩",
    layout="centered"
)

CLASS_NAMES = ['__background__', 'cat', 'chicken', 'cow', 'dog', 'dolphin', 'horse', 'iron golem', 'pig', 'rabbit', 'sheep', 'villager'] 

@st.cache_resource
def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))
    
    checkpoint = torch.load('minecraft_model_finetuned.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device

def predict_image(model, device, image, threshold):
    image_np = np.array(image)
    
    img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    img_draw = image_np.copy()
    
    count_dict = {name: 0 for name in CLASS_NAMES if name != '__background__'}

    for i in range(len(scores)):
        if scores[i] > threshold:
            x_min, y_min, x_max, y_max = boxes[i].astype(int)
            label_id = labels[i]
            score = scores[i]
            class_name = CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else str(label_id)
            
            if class_name in count_dict:
                count_dict[class_name] += 1
            
            cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            text = f"{class_name} {score:.0%}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_draw, (x_min, y_min - 20), (x_min + w, y_min), (0, 255, 0), -1)
            cv2.putText(img_draw, text, (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    return img_draw, count_dict

st.title("Minecraft Object Detection")
st.write("Faça upload de uma screenshot do jogo.")

st.sidebar.header("Configurações")
confidence_threshold = st.sidebar.slider("Confiança mínima", 0.0, 1.0, 0.7, 0.05)

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    
    #if st.button("Detectar entidades"):
    with st.spinner('Processando...'):
        try:
            model, device = load_model()
            
            result_image, stats = predict_image(model, device, image, confidence_threshold)
            
            with col2:
                st.subheader("Resultado")
                st.image(result_image, use_container_width=True)
            
            st.success("Detecção concluída!")
            st.write("### Entidades encontradas:")
            
            stat_cols = st.columns(len(stats))
            for i, (mob, count) in enumerate(stats.items()):
                stat_cols[i].metric(label=mob, value=count)
                
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
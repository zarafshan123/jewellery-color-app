import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

# -------------------------
# 1. Extract Dominant Colors
# -------------------------
def extract_color(image_path, clusters=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_data = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=clusters)
    labels = kmeans.fit_predict(pixel_data)
    colors = kmeans.cluster_centers_.astype(int)

    # Count percentage of each color
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100

    return colors, percentages


# -------------------------
# 2. Jewelry Recommendation
# -------------------------
def recommend_jewelry(colors):
    recommendations = []

    for r, g, b in colors:
        if r > g and r > b:
            recommendations.append(("Gold", "🟡 Gold jewelry suits this warm tone."))
        elif b > r and b > g:
            recommendations.append(("Silver", "⚪ Silver jewelry suits this cool tone."))
        else:
            recommendations.append(("Rose Gold", "🌸 Rose gold suits this balanced tone."))
    
    return recommendations


# -------------------------
# 3. Streamlit Premium UI
# -------------------------

st.set_page_config(page_title="Jewelry Color Analyzer", layout="wide")

st.title("💍 AI-Powered Jewelry Color Analyzer")
st.write("Upload your photo and get personalized jewelry recommendations based on your color tones.")

uploaded = st.file_uploader("📸 Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns([1,1.2])

    with col1:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        temp_path = "temp.jpg"
        img.save(temp_path)

    with col2:
        colors, percentages = extract_color(temp_path)

        st.subheader("🎨 Dominant Colors & Percentages")

        for idx, color in enumerate(colors):
            r, g, b = color
            hex_color = '#%02x%02x%02x' % (r, g, b)
            pct = round(percentages[idx], 2)

            st.markdown(
                f"""
                <div style='display:flex;align-items:center;margin-bottom:8px'>
                    <div style='width:50px;height:25px;background:{hex_color};border-radius:5px;margin-right:10px'></div>
                    <span style='font-size:16px;color:#333;'>RGB: {color.tolist()} — <b>{pct}%</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("💡 Jewelry Recommendations")

        rec = recommend_jewelry(colors)

        for metal, text in rec:
            if metal == "Gold":
                st.success(f"🟡 {text}")
            elif metal == "Silver":
                st.info(f"⚪ {text}")
            else:
                st.warning(f"🌸 {text}")

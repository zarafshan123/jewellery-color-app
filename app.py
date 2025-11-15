import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------
# 1. Extract Dominant Colors WITHOUT CV2
# ---------------------------------------------------
def extract_color_pil(image, clusters=3):
    img = image.resize((200, 200))  # reduce size for speed
    pixel_data = np.array(img).reshape(-1, 3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=clusters)
    labels = kmeans.fit_predict(pixel_data)
    colors = kmeans.cluster_centers_.astype(int)

    # Count cluster percentage
    counts = np.bincount(labels)
    percentages = (counts / len(labels)) * 100

    return colors, percentages


# ---------------------------------------------------
# 2. Jewellery Recommendation
# ---------------------------------------------------
def recommend_jewelry(colors):
    recommendations = []

    for r, g, b in colors:
        if r > g and r > b:
            recommendations.append(("Gold", "🟡 Gold jewelry enhances warm tones."))
        elif b > r and b > g:
            recommendations.append(("Silver", "⚪ Silver jewelry enhances cool tones."))
        else:
            recommendations.append(("Rose Gold", "🌸 Rose gold suits this balanced tone."))

    return recommendations


# ---------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Jewellery Color Analyzer", layout="wide")

st.title("💍 AI-Powered Jewellery Color Analyzer")
st.write("Upload an image and get personalized jewellery suggestions based on dominant colors.")

uploaded_img = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        colors, percentages = extract_color_pil(img)

        st.subheader("🎨 Dominant Colors")

        for idx, color in enumerate(colors):
            r, g, b = color
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            pct = round(percentages[idx], 2)

            st.markdown(
                f"""
                <div style='display:flex;align-items:center;margin-bottom:10px;'>
                    <div style='width:60px;height:30px;background:{hex_color};
                                border-radius:5px;border:1px solid #ccc;margin-right:12px;'></div>
                    <span style='font-size:16px;'>RGB: {color.tolist()} — <b>{pct}%</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("💡 Recommended Jewellery")

        rec = recommend_jewelry(colors)

        for metal, message in rec:
            if metal == "Gold":
                st.success(message)
            elif metal == "Silver":
                st.info(message)
            else:
                st.warning(message)

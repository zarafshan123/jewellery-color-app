# app.py (updated)
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jewellery Color Analyzer", layout="wide")
st.title("💍 AI-Powered Jewellery Color Analyzer")
st.write("Upload up to 3 pictures (hand, ear, neck). App aggregates colors and gives 3 unique jewellery suggestions.")

LABELS = ["Gold", "Silver", "Rose Gold", "Platinum", "Bronze"]  # fallback order for filling

RECOMMENDATION_DB = {
    "Gold": ["Gold Necklace (G1)", "Gold Ring (G2)", "Gold Earrings (G3)"],
    "Silver": ["Silver Bracelet (S1)", "Silver Ring (S2)", "Silver Studs (S3)"],
    "Rose Gold": ["Rose Gold Pendant (R1)", "Rose Gold Ring (R2)", "Rose Gold Hoop (R3)"],
    "Platinum": ["Platinum Chain (P1)", "Platinum Studs (P2)"],
    "Bronze": ["Bronze Bracelet (B1)"]
}

# ----------------- COLOR EXTRACTION -----------------
def extract_color_from_pixel_array(pixel_array, clusters=3):
    """
    pixel_array: numpy array shape (N,3) of RGB pixels (0-255)
    returns: colors (clusters x 3 ints), percentages (clusters,)
    """
    # KMeans
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_array)
    colors = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(labels, minlength=clusters)
    percentages = (counts / counts.sum()) * 100
    return colors, percentages

def combine_and_extract(images, clusters=3):
    """
    images: list of PIL Images (RGB)
    Combine their pixels and run clustering once for aggregated result.
    """
    all_pixels = []
    for img in images:
        # resize a bit to speed up (keeps color distribution)
        img_small = img.resize((200, 200))
        arr = np.array(img_small).reshape(-1, 3)
        all_pixels.append(arr)
    if not all_pixels:
        return np.array([]), np.array([])
    combined = np.vstack(all_pixels)
    return extract_color_from_pixel_array(combined, clusters=clusters)

# ----------------- MAPPING COLORS TO METAL -----------------
def color_to_metal(rgb):
    r, g, b = rgb
    # simple heuristic: dominant channel -> metal
    if r > g and r > b:
        return "Gold"
    elif b > r and b > g:
        return "Silver"
    else:
        # if green or balanced, assume rose gold (pinkish) as middle ground
        return "Rose Gold"

def recommend_unique_metals(colors, percentages, top_k=3):
    """
    colors: array of cluster colors
    percentages: corresponding percentages
    returns: list of up to top_k unique metal labels in order of cluster importance
    """
    if len(colors) == 0:
        return []

    # sort clusters by percentage descending
    idx_sorted = np.argsort(percentages)[::-1]
    metals = []
    for idx in idx_sorted:
        metal = color_to_metal(colors[idx])
        if metal not in metals:
            metals.append(metal)
        if len(metals) >= top_k:
            break

    # fill remaining with LABELS order (avoid duplicates)
    for lab in LABELS:
        if len(metals) >= top_k:
            break
        if lab not in metals:
            metals.append(lab)

    return metals[:top_k]

# ----------------- STREAMLIT UI -----------------
uploaded_files = st.file_uploader("📸 Upload up to 3 images (hand, ear, neck)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    files = uploaded_files[:3]  # use first 3 only
    st.write(f"Processing {len(files)} image(s)...")
    images = []
    cols = st.columns(len(files))
    for i, f in enumerate(files):
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images.append(img)
            cols[i].image(img, caption=f.name, use_column_width=True)
        except Exception as e:
            st.error(f"Could not open {f.name}: {e}")

    if images:
        # aggregated dominant colors
        clusters = 3
        colors, percentages = combine_and_extract(images, clusters=clusters)

        st.subheader("🎨 Aggregated Dominant Colors")
        for idx in np.argsort(percentages)[::-1]:
            r, g, b = colors[idx]
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            pct = round(percentages[idx], 2)
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;margin-bottom:8px;'>
                    <div style='width:60px;height:30px;background:{hex_color};
                                border-radius:5px;border:1px solid #ccc;margin-right:12px;'></div>
                    <span style='font-size:15px;'>RGB: [{r}, {g}, {b}] — <b>{pct}%</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # final unique metal recommendations
        top_metals = recommend_unique_metals(colors, percentages, top_k=3)

        st.subheader("💡 Recommended Jewellery (unique)")
        final_items = []
        for m in top_metals:
            # pick first unused item for each metal
            for item in RECOMMENDATION_DB.get(m, []):
                if item not in final_items:
                    final_items.append(item)
                    break

        # if still less than 3, fill from DB
        if len(final_items) < 3:
            for lab in LABELS:
                for item in RECOMMENDATION_DB.get(lab, []):
                    if item not in final_items:
                        final_items.append(item)
                    if len(final_items) >= 3:
                        break
                if len(final_items) >= 3:
                    break

        for i, rec in enumerate(final_items[:3], start=1):
            st.write(f"{i}. {rec}")

else:
    st.info("Please upload 1 to 3 images (hand, ear, neck) to get recommendations.")

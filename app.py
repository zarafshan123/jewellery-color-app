# app.py (body-part aware recommendations)
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jewellery Color Analyzer", layout="wide")
st.title("💍 AI-Powered Jewellery Color Analyzer")
st.write("Upload up to 3 pictures (hand, ear, neck). Mark each image's body part so recommendations match the part.")

LABELS = ["Gold", "Silver", "Rose Gold", "Platinum", "Bronze"]  # fallback order

# RECOMMENDATION_DB now nested by metal -> body_part -> list of items
RECOMMENDATION_DB = {
    "Gold": {
        "hand": ["Gold Ring (G-Ring1)", "Gold Bracelet (G-Bracelet1)"],
        "ear": ["Gold Hoop Earrings (G-Ear1)", "Gold Studs (G-Ear2)"],
        "neck": ["Gold Necklace (G-Neck1)"]
    },
    "Silver": {
        "hand": ["Silver Ring (S-Ring1)", "Silver Bracelet (S-Bracelet1)"],
        "ear": ["Silver Drop Earrings (S-Ear1)"],
        "neck": ["Silver Pendant (S-Neck1)"]
    },
    "Rose Gold": {
        "hand": ["Rose Gold Ring (R-Ring1)"],
        "ear": ["Rose Gold Hoop (R-Ear1)"],
        "neck": ["Rose Gold Pendant (R-Neck1)"]
    },
    "Platinum": {
        "hand": ["Platinum Band (P-Ring1)"],
        "ear": ["Platinum Studs (P-Ear1)"],
        "neck": ["Platinum Chain (P-Neck1)"]
    },
    "Bronze": {
        "hand": ["Bronze Bracelet (B-Bracelet1)"],
        "ear": ["Bronze Hoops (B-Ear1)"],
        "neck": ["Bronze Choker (B-Neck1)"]
    }
}

PART_OPTIONS = ["hand", "ear", "neck", "unsure"]

# ----------------- COLOR EXTRACTION -----------------
def extract_color_from_pixel_array(pixel_array, clusters=3):
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_array)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels, minlength=clusters)
    percentages = (counts / counts.sum()) * 100
    return colors, percentages

def combine_and_extract(images, clusters=3):
    all_pixels = []
    for img in images:
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
    if r > g and r > b:
        return "Gold"
    elif b > r and b > g:
        return "Silver"
    else:
        return "Rose Gold"

def recommend_unique_metals(colors, percentages, top_k=3):
    if len(colors) == 0:
        return []
    idx_sorted = np.argsort(percentages)[::-1]
    metals = []
    for idx in idx_sorted:
        metal = color_to_metal(colors[idx])
        if metal not in metals:
            metals.append(metal)
        if len(metals) >= top_k:
            break
    for lab in LABELS:
        if len(metals) >= top_k:
            break
        if lab not in metals:
            metals.append(lab)
    return metals[:top_k]

# ----------------- BODY-PART AWARE RECOMMENDER -----------------
def get_recommendations_for_parts(metals, parts, top_n=3):
    """
    metals: list of predicted metals in order of importance
    parts: set/list of body parts user selected (e.g., {'hand'}, or {'hand','neck'})
    returns list of up to top_n unique item strings
    """
    final_items = []

    # Priority 1: for each metal in order, try to pick items matching the selected parts
    for metal in metals:
        for part in parts:
            # skip 'unsure' as a part when trying to select specific part items
            if part == "unsure":
                continue
            items = RECOMMENDATION_DB.get(metal, {}).get(part, [])
            for it in items:
                if it not in final_items:
                    final_items.append(it)
                    break  # take only one per (metal,part) in this pass
            if len(final_items) >= top_n:
                return final_items[:top_n]

    # Priority 2: if user marked 'unsure' or parts empty, allow any part items for metals
    for metal in metals:
        for part, items in RECOMMENDATION_DB.get(metal, {}).items():
            for it in items:
                if it not in final_items:
                    final_items.append(it)
                if len(final_items) >= top_n:
                    return final_items[:top_n]

    # Priority 3: fallback - fill from any metal/part not used yet
    for metal in LABELS:
        for part_items in RECOMMENDATION_DB.get(metal, {}).values():
            for it in part_items:
                if it not in final_items:
                    final_items.append(it)
                if len(final_items) >= top_n:
                    return final_items[:top_n]

    return final_items[:top_n]

# ----------------- STREAMLIT UI -----------------
uploaded_files = st.file_uploader("📸 Upload up to 3 images (hand, ear, neck)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    files = uploaded_files[:3]
    st.write(f"Processing {len(files)} image(s)...")

    images = []
    selected_parts = []  # will store chosen part per image
    cols = st.columns(len(files))
    for i, f in enumerate(files):
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images.append(img)
            cols[i].image(img, caption=f.name, use_column_width=True)
            # let user label the image
            part = cols[i].selectbox(f"Select body part for {f.name}", PART_OPTIONS, index=0, key=f"part_{i}")
            selected_parts.append(part)
        except Exception as e:
            st.error(f"Could not open {f.name}: {e}")

    # Determine the set of parts to use for filtering (ignore 'unsure' if other parts exist)
    parts_set = set([p for p in selected_parts if p != "unsure"])
    if not parts_set:
        # if all marked 'unsure' or none selected, keep 'unsure' to allow any part
        parts_set = set(selected_parts) if selected_parts else set(["unsure"])

    if images:
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

        top_metals = recommend_unique_metals(colors, percentages, top_k=3)

        st.subheader("💡 Recommended Jewellery (based on selected body parts)")
        st.write(f"Using parts: {', '.join(parts_set)}")
        final_recs = get_recommendations_for_parts(top_metals, parts_set, top_n=3)

        if final_recs:
            for i, rec in enumerate(final_recs, start=1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No matching items found for selected parts — showing general recommendations.")
            # fallback general items
            fallback = get_recommendations_for_parts(top_metals, set(["unsure"]), top_n=3)
            for i, rec in enumerate(fallback, start=1):
                st.write(f"{i}. {rec}")

else:
    st.info("Please upload 1 to 3 images (hand, ear, neck) and mark each image's body part to get matching recommendations.")

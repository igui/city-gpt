import streamlit as st
import time
from city_gpt_utils import (
    load_model, 
    build_prompt, 
    generate_names,
    get_country_options,
    get_feature_class_options,
    get_feature_code_options
)

# App configuration
st.set_page_config(
    page_title="City GPT Generator",
    page_icon="🏙️",
    layout="centered"
)

# Set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🏙️ City GPT Name Generator")
st.markdown("Generate realistic geographical names using a character-level Transformer trained on 13.4 million GeoNames records.")

# Attempt to load model early to show error if not trained
try:
    with st.spinner("Loading model..."):
        load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Generation Settings")

# Dropdowns for inputs
countries = ["Random"] + get_country_options()
selected_country = st.sidebar.selectbox("Country Code", countries, index=1) # Default to US

feature_classes = {"Random": "Random"}
feature_classes.update(get_feature_class_options())
selected_class_key = st.sidebar.selectbox(
    "Feature Class", 
    list(feature_classes.keys()), 
    format_func=lambda x: feature_classes[x],
    index=1 # Default to P
)

feature_codes = ["Random"] + get_feature_code_options()
selected_code = st.sidebar.selectbox("Feature Code", feature_codes, index=1) # Default to PPL

name_prefix = st.sidebar.text_input("Name Prefix (optional)", placeholder="e.g. New")

# Generation parameters
st.sidebar.divider()
num_names = st.sidebar.selectbox("Number of names", [1, 10, 50], index=1)

with st.sidebar.expander("Advanced Settings"):
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    top_k = st.number_input("Top K", min_value=0, value=0, help="0 means no Top-K filtering")

if top_k == 0:
    top_k = None

# Build prompt
country_arg = "" if selected_country == "Random" else selected_country
class_arg = "" if selected_class_key == "Random" else selected_class_key
code_arg = "" if selected_code == "Random" else selected_code

prompt = build_prompt(country_arg, class_arg, code_arg, name_prefix)

st.write("### Current Prompt")
st.code(prompt if prompt else "<empty prompt - fully random>", language="text")

# Generate button
if st.button("Generate Names", type="primary"):
    with st.spinner(f"Generating {num_names} names..."):
        start_time = time.time()
        
        try:
            results = generate_names(
                prompt_str=prompt,
                num_names=num_names,
                temperature=temperature,
                top_k=top_k
            )
            
            elapsed = time.time() - start_time
            
            st.success(f"Generated {num_names} names in {elapsed:.2f} seconds.")
            
            # Display results in a nice format
            st.write("### Results")
            for i, name in enumerate(results, 1):
                full_name = f"{name_prefix}{name}" if name_prefix else name
                st.markdown(f"**{i}.** {full_name}")
                
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")

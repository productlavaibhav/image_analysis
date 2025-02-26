import streamlit as st
import os
import io
import json
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Function to setup API credentials
def setup_credentials():
    # For Google Vision API
    if 'GOOGLE_CREDENTIALS' in st.secrets:
        # Use the provided secrets
        credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    else:
        # Look for credentials file
        try:
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            vision_client = vision.ImageAnnotatorClient()
            st.success("Google Vision API credentials loaded successfully.")
        except Exception as e:
            st.error(f"Error loading Google Vision API credentials: {e}")
            vision_client = None
    
    # For OpenAI API
    if 'OPENAI_API_KEY' in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if not openai.api_key:
            st.warning("OpenAI API key not found.")
    
    return vision_client

# Function to analyze image with Google Vision API
def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        
        # Perform different types of detection
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        
        # Extract results
        results = {
            "labels": [{"description": label.description, "score": label.score} 
                      for label in label_detection.label_annotations],
            "text": [{"description": text.description, "confidence": text.confidence if hasattr(text, 'confidence') else None}
                    for text in text_detection.text_annotations[:1]],  # Just get the full text
            "faces": [{"joy": face.joy_likelihood.name, 
                       "sorrow": face.sorrow_likelihood.name,
                       "anger": face.anger_likelihood.name,
                       "surprise": face.surprise_likelihood.name}
                     for face in face_detection.face_annotations],
            "logos": [{"description": logo.description} for logo in logo_detection.logo_annotations],
            "colors": [{"color": {"red": color.color.red, 
                                  "green": color.color.green, 
                                  "blue": color.color.blue},
                        "score": color.score,
                        "pixel_fraction": color.pixel_fraction}
                      for color in image_properties.image_properties_annotation.dominant_colors.colors[:5]]
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error analyzing image with Google Vision API: {e}")
        return None

# Function to encode image to base64 for OpenAI
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to analyze image with OpenAI
def analyze_with_openai(base64_image):
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",  # Make sure to use a model that supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return None

# Function to generate detailed prompt using OpenAI based on both analyses
def generate_prompt(vision_results, openai_description):
    try:
        # Prepare input for GPT
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        
        prompt = """
        Based on the provided thumbnail analyses, create a detailed description covering:
        1. What's happening in the thumbnail
        2. Category of video (e.g., gaming, tutorial, vlog)
        3. Theme and mood
        4. Colors used and their significance
        5. Elements and objects present
        6. Subject impressions (emotions, expressions)
        7. Text present and its purpose
        
        Create this as a structured, detailed prompt that could be used to recreate or understand the thumbnail's purpose.
        
        Analysis data:
        """
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a thumbnail analysis expert who can create detailed prompts based on image analysis data."},
                {"role": "user", "content": prompt + json.dumps(input_data, indent=2)}
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating prompt: {e}")
        return None

# Main app
def main():
    st.title("YouTube Thumbnail Analyzer")
    st.write("Upload a thumbnail to analyze it using Google Vision AI and OpenAI, and generate a detailed prompt.")
    
    # Initialize and check API clients
    vision_client = setup_credentials()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Thumbnail", use_column_width=True)
        
        # Convert to bytes for API processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        with st.spinner("Analyzing thumbnail..."):
            # Process with Google Vision API
            if vision_client:
                vision_results = analyze_with_vision(img_byte_arr, vision_client)
                
                # Process with OpenAI
                base64_image = encode_image(img_byte_arr)
                openai_description = analyze_with_openai(base64_image)
                
                # Show raw analysis results in expanders
                with col2:
                    with st.expander("Google Vision API Results"):
                        st.json(vision_results)
                    
                    with st.expander("OpenAI Description"):
                        st.write(openai_description)
                
                # Generate the detailed prompt
                if vision_results and openai_description:
                    st.subheader("Generated Thumbnail Analysis")
                    with st.spinner("Generating detailed prompt..."):
                        time.sleep(1)  # Small delay for better UX
                        detailed_prompt = generate_prompt(vision_results, openai_description)
                        if detailed_prompt:
                            st.markdown(detailed_prompt)
                            
                            # Add a download button for the prompt
                            st.download_button(
                                label="Download Analysis",
                                data=detailed_prompt,
                                file_name="thumbnail_analysis.txt",
                                mime="text/plain"
                            )
            else:
                st.error("Google Vision API client not initialized. Please check your credentials.")

if __name__ == "__main__":
    main()

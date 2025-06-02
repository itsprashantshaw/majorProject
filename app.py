import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image
import tempfile
import time
import google.generativeai as genai
import os
from gtts import gTTS
import base64

# Page config
st.set_page_config(page_title="ASL Sign Language Detector", layout="wide")

# Initialize session state
if 'text_output' not in st.session_state:
    st.session_state['text_output'] = []
if 'last_prediction_time' not in st.session_state:
    st.session_state['last_prediction_time'] = 0
if 'current_word' not in st.session_state:
    st.session_state['current_word'] = []
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = []
if 'current_text_input' not in st.session_state:
    st.session_state['current_text_input'] = ''
if 'complete_text_input' not in st.session_state:
    st.session_state['complete_text_input'] = ''
if 'ai_response' not in st.session_state:
    st.session_state['ai_response'] = ''
if 'ok_detected' not in st.session_state:
    st.session_state['ok_detected'] = False
if 'stop_capture' not in st.session_state:
    st.session_state['stop_capture'] = False
if 'api_status' not in st.session_state:
    st.session_state['api_status'] = ''
if 'is_muted' not in st.session_state:
    st.session_state['is_muted'] = False

# Gemini configuration
def get_api_key():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        return api_key
    except Exception:
        # If not found in secrets, try environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # If not found in environment, ask user
            api_key = st.text_input("Enter your Google API Key:", type="password")
            if not api_key:
                st.error("Please enter your Google API Key")
                return None
        return api_key
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     api_key = st.text_input("Enter your Google API Key:", type="password")
    #     if not api_key:
    #         st.error("Please enter your Google API Key")
    #         return None
    # return api_key

# Function to autoplay audio in streamlit
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    try:
        # Create a temporary file
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
            return temp_file
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

def get_ai_response(text):
    """Get response from Gemini for the given text"""
    try:
        st.session_state['api_status'] = "Sending text to AI..."
        api_key = get_api_key()
        if not api_key:
            st.session_state['api_status'] = "Error: API key not provided"
            return "Error: Please enter your Google API key."
        
        genai.configure(api_key=api_key)
        st.session_state['api_status'] = "Waiting for AI response..."
        
        # Create a proper prompt
        prompt = f"I am using sign language and just signed this: {text}. Please respond to this message appropriately."
        
        # Updated model name - use one of the current available models
        # Options: 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-pro'
        model = genai.GenerativeModel('gemini-1.5-flash')  # Fast and efficient
        response = model.generate_content(prompt)
        
        st.session_state['api_status'] = "Response received!"
        return response.text
    except Exception as e:
        st.session_state['api_status'] = f"Error: {str(e)}"
        return f"Error getting AI response: {str(e)}"

# Load the model
@st.cache_resource
def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
    return model

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J',
               10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S',
               19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:' ', 27:"OK", 28:"UNKNOWN"}

labels_dict1 = {0: 'Tell me about todays weather in West Bengal', 1: 'Tell me about todays trending news', 2: 'Will RCB win this year ipl ?', 3:'Tell me about todays special', 4:'Tell me a joke', 5:'Can you suggest me a healthy diet chart ?', 6:'G', 7:'H', 8:'I', 9:'J',
               10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S',
               19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:' ', 27:"OK", 28:"UNKNOWN"}

labels_dict2 = {0: 'What is the meaning of life', 1: 'Tell me a fun fact', 2: 'How can I learn sign language quickly?',3:'How can I improve my sleep?',
                4:'What’s the weather like today?',5:'Give me a blog idea on AI and education',6:'Tell me something positive',7:'Give me a motivational quote',
                8:'What are some good luck tips for interviews?',9:'Tell me something about space',10:'When is the next FIFA World Cup?',
                11:'Analyze the impact of social media on society',12:'Discuss the ethical implications of artificial intelligence',
                13:'Forecast the trends in electric vehicle technology',14:'Predict the future of virtual reality technology',
                15:'OK',16:'Estimate the population growth in India by 2030',17:'Propose a plan for reducing plastic waste'}

# Model and labels mapping
model_options = {
    "ASL Model": {"path": "./model.p", "labels": labels_dict},
    "Model 2": {"path": "./model.p", "labels": labels_dict1},
    "Model 3": {"path": "./conmodel.p", "labels": labels_dict2},

}

# Add Gemini model selection in sidebar
st.sidebar.title("Model Configuration")
selected_asl_model = st.sidebar.selectbox(
    "Select ASL Model",
    list(model_options.keys())
)

gemini_model_options = {
    "Gemini 1.5 Flash (Fast)": "gemini-1.5-flash",
    "Gemini 1.5 Pro (Balanced)": "gemini-1.5-pro",
    "Gemini 2.5 Pro (Advanced)": "gemini-2.5-pro"
}

selected_gemini_model = st.sidebar.selectbox(
    "Select Gemini Model",
    list(gemini_model_options.keys()),
    help="Flash is fastest, Pro is balanced, 2.5 Pro is most advanced but slower"
)

# Add mute toggle in sidebar
st.sidebar.title("Audio Settings")
if st.sidebar.toggle("Mute Text-to-Speech", value=st.session_state['is_muted']):
    st.session_state['is_muted'] = True
else:
    st.session_state['is_muted'] = False

# Load selected model
model = load_model(model_options[selected_asl_model]["path"])
current_labels = model_options[selected_asl_model]["labels"]

def add_to_sentence():
    if st.session_state['current_word']:
        word = ''.join(st.session_state['current_word'])
        st.session_state['sentence'].append(word)
        st.session_state['current_word'] = []
        # Update complete text when adding a word
        current_complete = st.session_state['complete_text_input']
        st.session_state['complete_text_input'] = (current_complete + ' ' + word).strip()
        st.session_state['current_text_input'] = ''

def get_ai_response_with_model(text):
    """Get response from Gemini for the given text using selected model"""
    try:
        st.session_state['api_status'] = "Sending text to AI..."
        api_key = get_api_key()
        if not api_key:
            st.session_state['api_status'] = "Error: API key not provided"
            return "Error: Please enter your Google API key."
        
        genai.configure(api_key=api_key)
        st.session_state['api_status'] = "Waiting for AI response..."
        
        # Create a proper prompt
        prompt = f"I am using sign language and just signed this: {text}. Please respond to this message appropriately."
        
        # Use the selected Gemini model
        model_name = gemini_model_options[selected_gemini_model]
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        st.session_state['api_status'] = "Response received!"
        response_text = response.text
        
        # Convert response to speech only if not muted
        if not st.session_state['is_muted']:
            st.session_state['api_status'] = "Converting response to speech..."
            audio_file = text_to_speech(response_text)
            if audio_file:
                autoplay_audio(audio_file)
                # Clean up the temporary file
                try:
                    os.remove(audio_file)
                except:
                    pass
        
        return response_text
    except Exception as e:
        st.session_state['api_status'] = f"Error: {str(e)}"
        # Fallback to a different model if the selected one fails
        if "not found" in str(e).lower() or "not supported" in str(e).lower():
            try:
                st.session_state['api_status'] = "Trying fallback model..."
                fallback_model = genai.GenerativeModel('gemini-1.5-flash')
                response = fallback_model.generate_content(prompt)
                st.session_state['api_status'] = "Response received (fallback model)!"
                response_text = response.text
                
                # Convert fallback response to speech only if not muted
                if not st.session_state['is_muted']:
                    st.session_state['api_status'] = "Converting response to speech..."
                    audio_file = text_to_speech(response_text)
                    if audio_file:
                        autoplay_audio(audio_file)
                        # Clean up the temporary file
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                
                return response_text
            except Exception as fallback_error:
                st.session_state['api_status'] = f"Error with fallback: {str(fallback_error)}"
                return f"Error getting AI response: {str(fallback_error)}"
        return f"Error getting AI response: {str(e)}"

def process_frame(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape
    
    results = hands.process(frame_rgb)
    predicted_character = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Process landmarks for prediction
            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
            # Make prediction using current_labels
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = current_labels[int(prediction[0])]
            
            # If OK is detected and we have text, get AI response
            if predicted_character == "OK" and not st.session_state['ok_detected']:
                st.session_state['ok_detected'] = True
                st.session_state['stop_capture'] = True
                current_word = ''.join(st.session_state['current_word'])
                if current_word:  # Only send if there's text to send
                    prompt = f"Please respond to this: {current_word}"
                    st.session_state['ai_response'] = get_ai_response_with_model(prompt)
            
            # Draw bounding box and prediction
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    
    return frame, predicted_character

# UI Layout
st.title("ASL Sign Language Detector")

# Sidebar controls
st.sidebar.title("Controls")
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Video"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Video Feed")
    if input_source == "Webcam":
        col1_1, col1_2 = st.columns([1, 1])
        with col1_1:
            run = st.button("Start Camera")
        with col1_2:
            if st.button("Stop"):
                st.session_state['stop_capture'] = True
                st.session_state['ok_detected'] = True

        if run:
            video_placeholder = st.empty()
            text_display = st.empty()
            cap = cv2.VideoCapture(0)
            
            # Add text input for real-time display
            with text_display:
                current_text = st.text_input(
                    "Current Detection",
                    value=st.session_state['current_text_input'],
                    key="realtime_input"
                )
            
            # Reset stop flag when starting
            st.session_state['stop_capture'] = False
            st.session_state['ok_detected'] = False
            
            while cap.isOpened() and not st.session_state['stop_capture']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, predicted_char = process_frame(frame)
                
                # Display frame
                video_placeholder.image(processed_frame, channels="BGR")
                
                # Add prediction to output if valid and enough time has passed
                current_time = time.time()
                if predicted_char and (current_time - st.session_state['last_prediction_time']) > 2.0:
                    if predicted_char == "OK":
                        st.session_state['stop_capture'] = True
                        if st.session_state['complete_text_input']:
                            st.session_state['ai_response'] = get_ai_response_with_model(st.session_state['complete_text_input'])
                    elif predicted_char != "UNKNOWN":  # Only add valid characters
                        st.session_state['current_word'].append(predicted_char)
                        st.session_state['last_prediction_time'] = current_time
                        
                        # Update the text input with current word
                        st.session_state['current_text_input'] = ''.join(st.session_state['current_word'])
                        
                        # Update real-time display
                        with text_display:
                            st.text_input(
                                "Current Detection",
                                value=st.session_state['current_text_input'],
                                key=f"realtime_input_{current_time}"
                            )
                    
                # Add small delay to reduce CPU usage
                time.sleep(0.01)
            
            cap.release()
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            video_placeholder = st.empty()
            text_display = st.empty()
            
            # Add text input for real-time display
            with text_display:
                current_text = st.text_input(
                    "Current Detection",
                    value=st.session_state['current_text_input'],
                    key="realtime_input_video"
                )
            
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, predicted_char = process_frame(frame)
                
                # Display frame
                video_placeholder.image(processed_frame, channels="BGR")
                
                # Add prediction to output if valid and enough time has passed
                current_time = time.time()
                if predicted_char and (current_time - st.session_state['last_prediction_time']) > 2.0:
                    if predicted_char != "OK":  # Only add non-OK characters
                        st.session_state['current_word'].append(predicted_char)
                        st.session_state['last_prediction_time'] = current_time
                        
                        # Update the text input with current word
                        st.session_state['current_text_input'] = ''.join(st.session_state['current_word'])
                        
                        # Update real-time display
                        with text_display:
                            st.text_input(
                                "Current Detection",
                                value=st.session_state['current_text_input'],
                                key=f"realtime_input_{current_time}"
                            )
                    
                # Add small delay to reduce CPU usage
                time.sleep(0.01)
                cv2.waitKey(1)
            
            cap.release()

with col2:
    st.header("Complete Translation")
    
    # Current word text field
    current_word = st.text_input(
        "Current Word",
        value=''.join(st.session_state['current_word']),
        key="current_word_input"
    )
    
    # Complete translation text field
    complete_translation = st.text_area(
        "Complete Translation",
        value=st.session_state['complete_text_input'],
        height=100
    )
    
    # API Status indicator
    if st.session_state['api_status']:
        st.info(st.session_state['api_status'])
    
    # Update session state when complete translation is edited
    if complete_translation != st.session_state['complete_text_input']:
        st.session_state['complete_text_input'] = complete_translation
        # Update sentence list based on the complete translation
        st.session_state['sentence'] = complete_translation.split() if complete_translation else []
    
    col2_1, col2_2, col2_3 = st.columns([1, 1, 1])
    with col2_1:
        if st.button("Add Space"):
            add_to_sentence()
    with col2_2:
        if st.button("Clear All"):
            st.session_state['sentence'] = []
            st.session_state['current_word'] = []
            st.session_state['current_text_input'] = ''
            st.session_state['complete_text_input'] = ''
            st.session_state['ai_response'] = ''
            st.session_state['ok_detected'] = False
            st.session_state['api_status'] = ''
    with col2_3:
        if st.button("Backspace"):
            if st.session_state['current_word']:
                st.session_state['current_word'].pop()
                st.session_state['current_text_input'] = ''.join(st.session_state['current_word'])
            elif st.session_state['sentence']:
                st.session_state['sentence'].pop()
                st.session_state['complete_text_input'] = ' '.join(st.session_state['sentence'])

    # Manual AI Response trigger
    if st.button("Get AI Response"):
        if st.session_state['complete_text_input']:
            st.session_state['ai_response'] = get_ai_response_with_model(st.session_state['complete_text_input'])
        else:
            st.warning("Please enter some text first")

    # Add AI Response section
    st.subheader("AI Response")
    st.text_area(
        "AI Response",
        value=st.session_state['ai_response'],
        height=150
    )

st.markdown("""
### Instructions:
1. Select ASL model and Gemini model from the sidebar
2. Select input source (Webcam or Upload Video)
3. For webcam: Click 'Start Camera' and show ASL signs
4. For video: Upload a video file containing ASL signs
5. View real-time detection and translation
6. Use 'Add Space' to separate words
7. Use 'Backspace' to delete last character
8. Use 'Clear All' to reset
9. Use 'Get AI Response' to manually trigger AI response

### Model Options:
- **Gemini 1.5 Flash**: Fastest responses, good for real-time interaction
- **Gemini 1.5 Pro**: Balanced speed and quality
- **Gemini 2.5 Pro**: Most advanced but slower responses
""")
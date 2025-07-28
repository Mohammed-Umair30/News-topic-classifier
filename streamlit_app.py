import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Set page config
st.set_page_config(
    page_title="📰 News Topic Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_dir = "./fine_tuned_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

# Prediction function
def predict_topic(text, model, tokenizer):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].numpy()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📰 News Topic Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Fine-tuned DistilBERT</p>', unsafe_allow_html=True)
    
    # Load model
    try:
        model, tokenizer = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown("""
        **Model Architecture:**
        - DistilBERT for sequence classification
        - Fine-tuned on AG News dataset
        - 4 topic categories
        
        **Training Data:**
        - 100 training samples (demo)
        - 20 test samples (demo)
        - Max sequence length: 32 tokens
        
        **Performance:**
        - Accuracy: 12.0%
        - F1-score: 2.57%
        - *Note: Limited training for demo*
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔍 Enter News Headline</h2>', unsafe_allow_html=True)
        
        # Text input
        user_input = st.text_area(
            "Type or paste a news headline here:",
            placeholder="e.g., Apple announces new iPhone with advanced AI features",
            height=100
        )
        
        # Example headlines
        st.markdown("**📝 Try these examples:**")
        examples = [
            "Apple announces new iPhone with advanced AI features",
            "Manchester United wins Premier League championship",
            "Stock market reaches new all-time high amid economic growth",
            "Scientists discover new exoplanet in habitable zone"
        ]
        
        example_cols = st.columns(2)
        for i, example in enumerate(examples):
            with example_cols[i % 2]:
                if st.button(f"📄 Example {i+1}", key=f"example_{i}"):
                    user_input = example
                    st.rerun()
        
        # Prediction button
        if st.button("🚀 Classify Topic", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("🔄 Analyzing headline..."):
                    # Make prediction
                    predicted_class, confidence, probabilities = predict_topic(user_input, model, tokenizer)
                    
                    # Class labels
                    class_labels = ["World", "Sports", "Business", "Sci/Tech"]
                    predicted_topic = class_labels[predicted_class]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-text">📰 {predicted_topic}</div>
                        <div class="confidence-text">Confidence: {confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown('<h3 class="sub-header">📊 Probability Distribution</h3>', unsafe_allow_html=True)
                    
                    # Create probability chart
                    prob_df = pd.DataFrame({
                        'Topic': class_labels,
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Topic', 
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='viridis',
                        title="Topic Classification Probabilities"
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        xaxis_title="News Topics",
                        yaxis_title="Probability"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.markdown('<h3 class="sub-header">📈 Detailed Results</h3>', unsafe_allow_html=True)
                    
                    results_cols = st.columns(4)
                    for i, (topic, prob) in enumerate(zip(class_labels, probabilities)):
                        with results_cols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{topic}</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {'#28a745' if i == predicted_class else '#6c757d'}">
                                    {prob:.2%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a news headline to classify.")
    
    with col2:
        st.markdown('<h2 class="sub-header">📚 Topic Categories</h2>', unsafe_allow_html=True)
        
        # Topic descriptions
        topics_info = {
            "🌍 World": "International news, politics, global events",
            "⚽ Sports": "Sports news, games, athletes, competitions",
            "💼 Business": "Financial news, markets, companies, economy",
            "🔬 Sci/Tech": "Science, technology, research, innovations"
        }
        
        for topic, description in topics_info.items():
            st.markdown(f"""
            <div class="info-box">
                <h4>{topic}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model performance
        st.markdown('<h2 class="sub-header">⚡ Model Performance</h2>', unsafe_allow_html=True)
        
        # Performance metrics
        metrics_data = {
            "Metric": ["Accuracy", "F1-Score", "Training Samples", "Test Samples"],
            "Value": ["12.0%", "2.57%", "100", "20"]
        }
        
        for metric, value in zip(metrics_data["Metric"], metrics_data["Value"]):
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #eee;">
                <span><strong>{metric}:</strong></span>
                <span>{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <p><strong>Note:</strong> This is a demonstration model trained on a small subset of data. 
            For production use, train on the full dataset with more epochs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏠 <strong>News Topic Classifier</strong> | Built with Streamlit & DistilBERT</p>
        <p>This is a demonstration of multimodal machine learning for text classification.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


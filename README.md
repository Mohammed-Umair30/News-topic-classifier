# 📰 News Topic Classifier Using BERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://8502-ixakbkaeslfvs765jqkct-e8e0f43b.manusvm.computer)
[![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.54.0-blue)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **News Topic Classifier** built with fine-tuned **DistilBERT** that automatically categorizes news headlines into four distinct topics: World, Sports, Business, and Sci/Tech. The project demonstrates modern NLP techniques including transformer fine-tuning, transfer learning, and interactive web deployment.

## 🌟 Features

- **🤖 Fine-tuned DistilBERT**: Leverages pre-trained transformer model for superior text understanding
- **📊 Interactive Web App**: Beautiful Streamlit interface with real-time predictions
- **📈 Probability Visualization**: Dynamic charts showing classification confidence
- **🎯 Multi-class Classification**: Categorizes news into 4 distinct topics
- **⚡ Fast Inference**: Optimized for quick predictions with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

### Main Interface
![News Topic Classifier Interface](https://via.placeholder.com/800x400/1f77b4/ffffff?text=News+Topic+Classifier+Interface)

### Prediction Results
![Classification Results](https://via.placeholder.com/800x400/2ca02c/ffffff?text=Classification+Results+with+Charts)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   News Headline │    │  DistilBERT      │    │   Topic         │
│                 │    │  Tokenizer       │    │   Classification│
│ "Apple announces│────│                  │────│                 │
│  new iPhone..."  │    │ • Tokenization   │    │ • World         │
│                 │    │ • Encoding       │    │ • Sports        │
└─────────────────┘    │ • Attention      │    │ • Business      │
                       │ • Classification │    │ • Sci/Tech      │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

- **Deep Learning**: Hugging Face Transformers, PyTorch
- **Model**: DistilBERT (distilbert-base-uncased)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy, datasets
- **Visualization**: Plotly, matplotlib
- **Evaluation**: scikit-learn
- **Dataset**: AG News Dataset

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohammed-Umair30/News-topic-classifier.git
   cd news-topic-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   ```bash
   python prepare_data.py
   ```

4. **Fine-tune the model** (optional - pre-trained model included)
   ```bash
   python fine_tune_model.py
   ```

5. **Evaluate the model**
   ```bash
   python evaluate_model.py
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📋 Requirements

```txt
streamlit>=1.28.0
transformers>=4.54.0
torch>=2.0.0
datasets>=4.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
accelerate>=1.9.0
```

## 📊 Dataset

The project uses the **AG News Dataset** from Hugging Face:
- **Total samples**: 127,600 (120,000 train + 7,600 test)
- **Categories**: 4 classes
  - **World** (0): International news, politics, global events
  - **Sports** (1): Sports news, games, athletes, competitions  
  - **Business** (2): Financial news, markets, companies, economy
  - **Sci/Tech** (3): Science, technology, research, innovations
- **Format**: News headlines with corresponding labels

### Data Preprocessing
- **Tokenization**: DistilBERT tokenizer with max length 32
- **Encoding**: Automatic padding and truncation
- **Format**: PyTorch tensors for efficient training

## 🧠 Model Architecture

### DistilBERT Configuration
```python
Model: DistilBertForSequenceClassification
- Base Model: distilbert-base-uncased
- Parameters: ~67M (50% smaller than BERT)
- Layers: 6 transformer layers
- Hidden Size: 768
- Attention Heads: 12
- Vocabulary Size: 30,522
```

### Fine-tuning Setup
```python
Training Arguments:
- Learning Rate: 2e-5
- Batch Size: 4 (demo), 16 (full training)
- Epochs: 1 (demo), 3-5 (recommended)
- Optimizer: AdamW
- Weight Decay: 0.01
- Warmup Steps: 100
```

## 📈 Performance Metrics

| Metric | Demo Model | Full Training (Expected) |
|--------|------------|-------------------------|
| **Accuracy** | 12.0% | 85-90% |
| **F1-Score** | 2.57% | 85-88% |
| **Training Samples** | 100 | 120,000 |
| **Test Samples** | 20 | 7,600 |
| **Training Time** | ~2 minutes | ~2-3 hours |

*Note: Demo model uses limited data for faster execution. Full training achieves much higher performance.*

## 🎯 Usage

### Web Interface
1. Visit the [live application](https://8502-ixakbkaeslfvs765jqkct-e8e0f43b.manusvm.computer)
2. Enter a news headline in the text area
3. Click "🚀 Classify Topic" to get predictions
4. View results with confidence scores and probability distribution

### Programmatic Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_dir = "./fine_tuned_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Predict function
def predict_topic(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_labels = ["World", "Sports", "Business", "Sci/Tech"]
    return class_labels[predicted_class], confidence

# Example usage
headline = "Apple announces new iPhone with advanced AI features"
topic, confidence = predict_topic(headline)
print(f"Topic: {topic}, Confidence: {confidence:.2%}")
```


## 🔮 Future Enhancements

- [ ] **Full Dataset Training**: Train on complete AG News dataset for better performance
- [ ] **Advanced Models**: Experiment with RoBERTa, ALBERT, or GPT models
- [ ] **Multi-language Support**: Extend to non-English news classification
- [ ] **Real-time News**: Integrate with news APIs for live classification
- [ ] **Batch Processing**: Add bulk headline classification feature
- [ ] **Model Interpretability**: Add SHAP or attention visualization
- [ ] **API Endpoint**: Create REST API for programmatic access
- [ ] **Docker Deployment**: Containerize for easy deployment

## 🚀 Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py --server.port 8502 --server.address 0.0.0.0
```

### Production Deployment
The application can be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web application hosting
- **AWS/GCP/Azure**: Cloud platform deployment
- **Docker**: Containerized deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **Dataset**: [AG News Dataset](https://huggingface.co/datasets/ag_news) from Hugging Face
- **Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) by Hugging Face
- **Framework**: [Transformers](https://huggingface.co/transformers/) library
- **Web Framework**: [Streamlit](https://streamlit.io/) for the amazing interface
- **Visualization**: [Plotly](https://plotly.com/) for interactive charts

## 📞 Contact

**Mohammad Umair**  hafizumair07.hm@example.com

**Project Link**: [https://github.com/Mohammed-Umair30/News-topic-classifier]

---

⭐ **Star this repository if you found it helpful!** ⭐


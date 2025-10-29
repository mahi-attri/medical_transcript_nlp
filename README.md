# Medical NLP Pipeline - Physician Notetaker 

A comprehensive AI system for medical transcription, NLP-based summarization, and sentiment analysis. This project implements state-of-the-art natural language processing techniques to extract key medical information from physician-patient conversations.

## Quick Start

### Use Jupyter Notebook

```Command Terminal
jupyter notebook medical_nlp_notebook.ipynb
```
Run cells sequentially to:
- See step-by-step processing
- Visualize results with charts
- Experiment with different inputs

## Installation

## Step 1: Install Dependencies

## Install all required packages
!pip install spacy transformers torch sentence-transformers scikit-learn nltk pandas matplotlib seaborn

## Download spaCy language model
python -m spacy download en_core_web_sm

## Download NLTK data
python -c "import nltk; nltk.download('punkt')"

## Features

### 1. **Named Entity Recognition (NER)**
- Extract **Symptoms** (neck pain, back pain, discomfort)
- Identify **Treatments** (physiotherapy, painkillers, X-rays)
- Detect **Diagnosis** (whiplash injury, strain)
- Capture **Prognosis** (full recovery, improvement timeline)
- Extract temporal information (dates, durations)

### 2. **Text Summarization**
- Convert lengthy transcripts into structured medical reports
- Generate JSON-formatted summaries with all key information
- Maintain medical accuracy and context

### 3. **Keyword Extraction**
- Identify important medical phrases
- Frequency-based and context-aware extraction
- Support for multi-word medical terms

### 4. **Sentiment Analysis**
- Classify patient emotional state: **Anxious**, **Neutral**, **Reassured**
- Use transformer-based models (DistilBERT)
- Provide confidence scores for each classification

### 5. **Intent Detection**
- Identify patient communication intent:
  - Seeking reassurance
  - Reporting symptoms
  - Expressing concern
  - Asking questions
  - Providing information
- Zero-shot classification for flexibility

### 6. **SOAP Note Generation** (Bonus)
- **S**ubjective: Patient's description of symptoms and history
- **O**bjective: Physical examination findings
- **A**ssessment: Diagnosis and severity
- **P**lan: Treatment plan and follow-up recommendations

## Architecture

```
Medical NLP Pipeline
│
├── Input Layer
│   └── Medical Transcript (Text)
│
├── Processing Layers
│   ├── Tokenization (NLTK)
│   ├── NER (spaCy + Rule-based)
│   ├── Sentiment Analysis (DistilBERT)
│   ├── Intent Classification (BART)
│   └── Text Generation (Custom Logic)
│
└── Output Layer
    ├── Structured Summary (JSON)
    ├── SOAP Note (JSON)
    └── Analysis Reports
```

### Component Breakdown

1. **spaCy (en_core_web_sm)**
   - Part-of-speech tagging
   - Named entity recognition
   - Noun chunk extraction
   - Dependency parsing

2. **DistilBERT**
   - Pre-trained sentiment analysis
   - Fine-tuned on SST-2 dataset
   - Fast inference with good accuracy

3. **BART-large-MNLI**
   - Zero-shot classification
   - Intent detection without training
   - Multi-label support

4. **Rule-based Extractors**
   - Medical keyword matching
   - Pattern recognition
   - Context-aware extraction


### Pre-trained NLP Models Used

#### 1. **spaCy - en_core_web_sm**
- **Purpose:** General NER and linguistic processing
- **Accuracy:** Good for general medical text
- **Why chosen:** Fast, lightweight, good baseline

#### 2. **DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)**
- **Purpose:** Sentiment analysis
- **Why chosen:** 
  - 60% faster than BERT
  - 40% smaller model size
  - Maintains 97% of BERT's performance
  - Good for medical sentiment with contextual understanding

#### 3. **BART-large-MNLI (facebook/bart-large-mnli)**
- **Purpose:** Zero-shot intent classification
- **Accuracy:** State-of-the-art on MNLI benchmark
- **Why chosen:**
  - No training data required
  - Flexible intent categories
  - Excellent at understanding medical context
  - Multi-label classification support


## Output Examples

### Structured Summary Output

### Structured Summary
```   "structured_summary": {
    "Patient_Name": "Jones",
    "Symptoms": [
      "some discomfort",
      "pain",
      "back pain",
      "painkillers",
      "the stiffness"
    ],
    "Diagnosis": "a whiplash injury",
    "Treatment": [
      "painkillers",
      "physiotherapy",
      "ten sessions"
    ],
    "Current_Status": "Improving",
    "Prognosis": "Given your progress, I'd expect you to make a full recovery within six months of the accident."
  },
```

### Sentiment Analysis
``` "sentiment_analysis": [
    {
      "text": "I'm doing better, but I still have some discomfort now and then.",
      "Sentiment": "Anxious",
      "Confidence": 0.996
    },
    {
      "text": "The first four weeks were rough. My neck and back pain were really bad.",
      "Sentiment": "Anxious",
      "Confidence": 1.0
    },
    {
      "text": "I'm a bit worried about my back pain, but I hope it gets better soon.",
      "Sentiment": "Reassured",
      "Confidence": 0.973
    },
    {
      "text": "That's great to hear. So, I don't need to worry about this affecting me in the future?",
      "Sentiment": "Reassured",
      "Confidence": 1.0
    }
```

### Intent Detection
``` "intent_detection": [
    {
      "text": "I'm doing better, but I still have some discomfort now and then.",
      "Intent": "Reporting symptoms",
      "Confidence": 0.497
    },
    {
      "text": "The first four weeks were rough. My neck and back pain were really bad.",
      "Intent": "Reporting symptoms",
      "Confidence": 0.45
    },
    {
      "text": "I'm a bit worried about my back pain, but I hope it gets better soon.",
      "Intent": "Expressing concern",
      "Confidence": 0.647
    },
    {
      "text": "That's great to hear. So, I don't need to worry about this affecting me in the future?",
      "Intent": "Seeking reassurance",
      "Confidence": 0.454
    }
```

### SOAP Note Generation
``` "soap_note": {
    "Subjective": {
      "Chief_Complaint": "some discomfort",
      "History_of_Present_Illness": "Physician: I understand you were in a car accident last September. Physician: What did you feel immediately after the accident? Patient: Yes, I went to Moss Bank Accident and Emergency."
    },
    "Objective": {
      "Physical_Exam": "Let's go ahead and do a physical examination to check your mobility and any lingering pain.",
      "Observations": "Patient appears in normal health"
    },
    "Assessment": {
      "Diagnosis": "a whiplash injury",
      "Severity": "Mild, improving"
    },
    "Plan": {
      "Treatment": "painkillers, physiotherapy, ten sessions",
      "Follow_Up": "Given your progress, I'd expect you to make a full recovery within six months of the accident."
    }
  }
}
```

## Future Improvements

### Short-term
1. **Medical Entity Linking**
   - Link entities to medical ontologies (SNOMED CT, ICD-10)
   - Standardize terminology

2. **Multi-language Support**
   - Add Spanish, French, German models
   - Translation capabilities

3. **Voice Input**
   - Real-time speech-to-text
   - Speaker diarization (identify who's speaking)

### Long-term
1. **Fine-tuned Medical Models**
   - Train on large medical corpus
   - Improve accuracy for medical-specific terms

2. **Interactive Dashboard**
   - Web interface for doctors
   - Real-time transcription and analysis
   - Export to EHR systems

3. **Privacy & Compliance**
   - HIPAA compliance
   - Data anonymization
   - Secure storage

4. **Active Learning**
   - Human-in-the-loop corrections
   - Continuous model improvement
   - Feedback mechanism

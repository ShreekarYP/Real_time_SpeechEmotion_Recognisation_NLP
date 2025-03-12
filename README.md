# 🗣️ NLP Speech Emotion Recognition 🔍  
A real-time speech emotion recognition system using **Python, NLP, Speech Recognition, and Machine Learning**. It captures speech, processes text, and classifies emotions using **TF-IDF, Count Vectorization, Naive Bayes, SVM, and an ensemble model**.  

## 🚀 Features  
✅ Real-time speech-to-text conversion using **Google Speech Recognition**  
✅ Text preprocessing with **stemming, lemmatization, and stopword removal**  
✅ Feature extraction using **TF-IDF & Count Vectorization**  
✅ Balances imbalanced datasets using **SMOTE (Synthetic Minority Over-sampling Technique)**  
✅ Trains and evaluates models: **Naive Bayes, SVM, and Voting Classifier**  
✅ Interactive mode for **live emotion detection**  

## 📌 Technologies Used  
- **Python** (NLP and Machine Learning)  
- **SpeechRecognition** (Google Speech API)  
- **NLTK** (Text preprocessing)  
- **Scikit-learn** (TF-IDF, Count Vectorizer, SVM, Naive Bayes)  
- **Imbalanced-learn** (SMOTE for data balancing)  

## 🏗️ Project Structure  
```
📂 NLP-Speech-Emotion-Recognition
 ┣ 📂 data                # Training dataset (categorized by emotion)
 ┣ 📜 main.py             # Main script to train and run the model
 ┣ 📜 model.py            # Model training, evaluation, and prediction logic
 ┣ 📜 requirements.txt    # Dependencies for the project
 ┗ 📜 README.md           # Project documentation
```

## 🔧 Installation & Setup  
1️⃣ Clone this repository:  
   ```bash
   git clone https://github.com/your-username/NLP-Speech-Emotion-Recognition.git
   cd NLP-Speech-Emotion-Recognition
   ```  
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3️⃣ Download NLTK resources:  
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```  
4️⃣ Run the script:  
   ```bash
   python main.py
   ```  

## 🧠 How it Works  
1️⃣ **Speech Input**: Records audio and converts it into text using **Google Speech Recognition**  
2️⃣ **Text Preprocessing**: Cleans and standardizes text with **stemming, lemmatization, and stopword removal**  
3️⃣ **Feature Extraction**: Converts text into numerical form using **TF-IDF & Count Vectorization**  
4️⃣ **Model Training**: Uses **Naive Bayes, SVM, and Voting Classifier** for emotion classification  
5️⃣ **Prediction**: Classifies the detected emotion and displays the result  

## 📊 Understanding TF-IDF  
- **TF (Term Frequency)** = (Occurrences of a word in a sentence) / (Total words in the sentence)  
- **IDF (Inverse Document Frequency)** = log(Total number of sentences / Number of sentences containing the word)  
- **TF-IDF** assigns importance to words, filtering out common words while highlighting key terms.  

## 🎤 Live Emotion Detection  
Run the script and speak into your microphone. The system will:  
✅ Convert speech to text  
✅ Analyze emotions in real time  
✅ Display the detected emotion  

  



 

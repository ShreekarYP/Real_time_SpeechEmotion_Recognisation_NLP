import os
import numpy as np
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class NLPSpeechEmotionRecognizer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.recognizer = sr.Recognizer()
        self.lemmatizer = WordNetLemmatizer()                              # Lemmatization ensures words have meaning
        self.stemmer = PorterStemmer()                                     # Stemming ensures words are reduced to their shortest possible form.
                                                                           # By combining both, you improve text standardization for machine learning models.
        self.stop_words = set(stopwords.words('english'))
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) #(Term Frequency- Inverse document freequency)
        self.count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000) #(BoW)
        self.model = None
    
    def preprocess_text(self, text):                                 #4 
        tokens = word_tokenize(text.lower())
        cleaned_tokens = [
            self.stemmer.stem(self.lemmatizer.lemmatize(token))
            for token in tokens if token.isalnum() and token not in self.stop_words
        ]
        return ' '.join(cleaned_tokens)                                 #returns #3
    
    def record_speech(self, duration=5):
        with sr.Microphone() as source:
            print("Listening... Please speak.")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=duration)
        
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Speech recognition service error")
        return None
    
    def prepare_training_data(self, data_path):                              #2    
        texts, labels = [], []
        for emotion in os.listdir(data_path):                              # finding emotions in the path
            emotion_path = os.path.join(data_path, emotion)
            if os.path.isdir(emotion_path):
                for file in os.listdir(emotion_path):
                    with open(os.path.join(emotion_path, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                    processed_text = self.preprocess_text(text)                    #3
                    texts.append(processed_text)
                    labels.append(emotion)
        return texts, labels                                                     #returns #1
    
    def train_model(self, data_path):                                                 
        texts, labels = self.prepare_training_data(data_path)                      #1
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        smote = SMOTE()
        X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train) #SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic data for underrepresented classes.
        
        nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
        grid_search = GridSearchCV(MultinomialNB(), nb_params, cv=5)
        grid_search.fit(X_train_tfidf, y_train)
        best_nb = grid_search.best_estimator_
        
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(X_train_tfidf, y_train)
        
        ensemble_model = VotingClassifier(
            estimators=[('nb', best_nb), ('svm', svm_model)], voting='soft'
        )
        ensemble_model.fit(X_train_tfidf, y_train)
        
        nb_pred = best_nb.predict(X_test_tfidf)
        svm_pred = svm_model.predict(X_test_tfidf)
        ensemble_pred = ensemble_model.predict(X_test_tfidf)
        
        print("\nNaive Bayes Model Performance:")
        print(classification_report(y_test, nb_pred))
        
        print("\nSVM Model Performance:")
        print(classification_report(y_test, svm_pred))
        
        print("\nEnsemble Model Performance:")
        print(classification_report(y_test, ensemble_pred))
        
        self.model = ensemble_model
    
    def predict_emotion(self, text):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        return self.model.predict(vectorized_text)[0]
    
    def interactive_emotion_recognition(self):
        while True:
            try:
                speech_text = self.record_speech()
                if speech_text:
                    emotion = self.predict_emotion(speech_text)
                    print(f"Transcribed Text: {speech_text}")
                    print(f"Detected Emotion: {emotion}")
                
                if input("\nContinue? (y/n): ").lower() != 'y':
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

def main():
    TRAINING_DATA_PATH = "C:\\Users\\shree\\Downloads\\speech-emotion-recognition\\data"
    
    emotion_recognizer = NLPSpeechEmotionRecognizer()
    print("Training NLP-based Emotion Recognition Model...")
    emotion_recognizer.train_model(TRAINING_DATA_PATH)                  #0
    
    print("\nStarting Interactive Emotion Recognition...")
    emotion_recognizer.interactive_emotion_recognition()

if __name__ == "__main__":
    main()



    #-------------------------------------------------------------------------------------------------------------------------------------------
  
        
  #TF= (number of rep of word in the sentence/total number of words in the sentence)
  # IDF=log(number of sentences/total number of words containing sentences)
  # 
  # TF*IDF for each words in the all the sentences          

 
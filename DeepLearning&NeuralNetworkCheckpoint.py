import streamlit as st
import pyaudio
import speech_recognition as sr
import os
from deepgram import DeepgramClient,PrerecordedOptions,FileSource
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import wave

#directory = r'D:\\majed\\Data Science\\deployment\\checkpoint\\VoiceCheckpoint\\Speech_recognition'

#os.chdir(directory)

DEEPGRAM_API_KEY='68be8412d0aa265e6b0a02da9a415c0e8f38e355'



def transcribe_speech(Language) :

    # Initialisation de la classe de reconnaissance
    r = sr.Recognizer()
    # Lecture du microphone comme source

    with sr.Microphone() as source :

        st.info("Parlez maintenant..." )

        # écoute la parole et la stocke dans la variable audio_text

        audio_text = r.listen(source)

        st.info("Transcription..." )

        try :

            # utiliser la reconnaissance vocale de Google

            text = r.recognize_google(audio_text,language=Language)

            return text

        except Exception as e:
            print(f"Exception: {e}")  

            return "Désolé, je n'ai pas compris." 


def deepgram(Language):
    
    r = sr.Recognizer()
    dg= DeepgramClient(DEEPGRAM_API_KEY)
    
    
    try:  
        with sr.Microphone() as source :

            st.info("Parlez maintenant..." )

            # écoute la parole et la stocke dans la variable audio_text

            audio_text = r.listen(source)

            st.info("Transcription..." )

            with open("output.wav", "wb") as f: 
                f.write(audio_text.get_wav_data())

            with open("output.wav", "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }
        
            #STEP 2: Configure Deepgram options for audio analysis
            options = PrerecordedOptions(
            model="nova-2",
            language=Language,
            smart_format=True,
            )

            # STEP 3: Call the transcribe_file method with the text payload and options
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

            return response.to_json(indent=4)
        
    except Exception as e:
        print(f"Exception: {e}")    
        return "Désolé, je n'ai pas compris."
    



df0=pd.read_csv('DataScience QA.csv')
df1=pd.read_csv('DataScienceBasics_QandA - Sheet1.csv')
df1.drop('Id',axis=1,inplace=True)

df=pd.concat([df0,df1],axis=0)
df.reset_index(inplace=True)
df.drop_duplicates(inplace=True)
dfQ=df['Question']
dfA=df['Answer']
sentences=dfQ.apply(lambda x:sent_tokenize(x))
sentences=sentences.to_list()

def preprocess(sentence) :

    # Tokenize the sentence into words (Tokenisation de la phrase en mots)

    words = word_tokenize(sentence)

    # Suppression des mots vides et de la ponctuation

    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]

    # Lemmatisation des mots

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in words]

    return words


sentences = [str(sentence) for sentence in sentences]

# Prétraitement de chaque phrase du texte

corpus = [preprocess(sentence) for sentence in sentences]

# Définir une fonction pour trouver la phrase la plus pertinente en fonction d'une requête

def get_most_relevant_sentence(query) :

    # Prétraitement de la requête

    query = preprocess(query)

    # Calcule la similarité entre la requête et chaque phrase du texte

    index=0
    max_similarity = 0

    most_relevant_sentence = ""

    for sentence in corpus :

        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))

        if similarity > max_similarity :

            index=corpus.index(sentence)
            max_similarity = similarity

            most_relevant_sentence = sentence
    if max_similarity > 0.2:
        most_relevant_sentence=most_relevant_sentence
        
    else:
        most_relevant_sentence=''

    return most_relevant_sentence,index
        

def chatbot(question) :

    # Trouver la phrase la plus pertinente

    phrase_la_plus_pertinente,index = get_most_relevant_sentence(question)


    if phrase_la_plus_pertinente!='':
        
        question_answer=df.loc[[index]].values.tolist()

        Question=question_answer[0][1]

        Answer=question_answer[0][2]
    else:
        Question='?'
        Answer='Sorry, it seems that I have no answer. Please try new query'

    # Retourne la réponse

    return Question,Answer







def main() :

    st.title("Chatbot application with text and voice Input " )

    st.write("Bonjour ! Je suis un chatbot. Demandez-moi n'importe quoi sur le sujet de Data science. Vous Pouvez ecrire votre question ou la donner par microphone")

# Obtenir la question de l'utilisateur

    question = st.text_input("You:" )

    # Créer un bouton pour soumettre la question

    text=''

    api_to_use=1
    Language='en-US'
    
    
    API = st.radio(
        'Choisir l''API a utiliser:',
        ('Speech Recognizer','Deepgram'))
    if API:
        if API=='Deepgram':
            api_to_use=2



    st.write("Cliquez sur le microphone pour commencer à parler:" )



    # ajouter un bouton pour déclencher la reconnaissance vocale

    if st.button("Start Recording"):

        if api_to_use==1:
            text = transcribe_speech(Language)
        elif api_to_use==2:
            text = deepgram(Language)


        st.write("Transcription : " , text)
    

    if st.button("Submit"):

        
        if text=='' or text=='Sorry, it seems that I have no answer. Please try new query':
            # Appeler la fonction chatbot avec la question et afficher la réponse
            Question,Answer = chatbot(question)
        else:
            Question,Answer = chatbot(text)

        st.write("Chatbot : Your question is: \n  "  + Question)
        st.write("Chatbot : The answer is:  \n     "  + Answer)

if __name__ == "__main__" :

    main()
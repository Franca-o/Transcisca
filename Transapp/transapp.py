
import gradio as gr
import whisper

import os
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader
from pytube import YouTube
import webbrowser
from bs4 import BeautifulSoup


os.environ["OPENAI_API_KEY"]="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"
model= whisper.load_model("base")
embeddings = OpenAIEmbeddings()
transcription=""
import gradio as gr

theme = gr.themes.Base(
    primary_hue="stone",
    neutral_hue="neutral",
    
)
with gr.Blocks(theme=theme) as demo:
   
#transcription section
#transcripts are stored in a file named "OutputA.txt"
    def transcribe(file2,file1,link):
        model= whisper.load_model("base")
#function to transcribe video file
        if file1:
            result= model.transcribe(file1)
            transcription= result["text"]
            print(result["text"])
            folder_path = "docs"  # Path to the existing "docs" folder
            file_name = "outputA"
# Combine the folder path and file name to get the file path
            file_path = os.path.join(folder_path, file_name +".txt")
            with open(file_path, "w") as file:
                file.write(transcription)
        
            return transcription
#function to transcribe audio file
        elif file2:
            result= model.transcribe(file2)
            transcription= result["text"]
            print(result["text"])
            folder_path = "docs"  # Path to the existing "docs" folder
            file_name = "outputA"
            # Combine the folder path and file name to get the file path
            file_path = os.path.join(folder_path, file_name +".txt")
            with open(file_path, "w") as file:
                file.write(transcription)
        
            return transcription
#function to transcribe youtube video
        if link:
#converts link passed a string into and html code and extracts the href from the html file
                html_code = f'<html><body><a href="{link}" target="_blank">{link}</a></body></html>'
                html_file = "hyperlink.html"
        
                with open(html_file, 'w') as file:
                    file.write(html_code)

                webbrowser.open(html_file)
                print(html_code)

                with open(html_file, 'r') as file:
                    html_content = file.read()

                soup = BeautifulSoup(html_content, 'html.parser')
                hyperlink_tags = soup.find_all('a')

                if hyperlink_tags:
 # Extract the href attribute from the first <a> tag
                    href = hyperlink_tags[0].get('href')
                   
#obtain the video from the youtube link for transcription
                video_URL = href
                destination = "."
                final_filename="ytaudio"
                video = YouTube(video_URL)

# Convert video to Audio
                audio = video.streams.filter(only_audio=True).first()
# Save to destination
                output = audio.download(output_path = destination)

                _, ext = os.path.splitext(output)
                new_file = final_filename + '.mp3'
# Delete the existing mp3 file if it exists
# (this is to overwrite the content of the current file for every transcription)
                audio_file = os.path.join(destination, new_file)
                if os.path.exists(audio_file):
                        os.remove(audio_file)

# Change the name of the file
                os.rename(output, new_file)
                audio_file = "ytaudio.mp3"

                result = model.transcribe(audio_file)
                answer= result["text"] 
                folder_path = "docs"  # Path to the existing "docs" folder
                file_name = "outputA"
 # Combine the folder path and file name to get the file path
                file_path = os.path.join(folder_path, file_name +".txt")
                with open(file_path, "w", encoding="utf-8") as file:
                            file.write(answer)
                return answer
#function to read PDF uploaded              
    def ret (file,text):
        if file:
            with open(file.name, 'rb') as f:
                reader = PdfReader(f)
                raw_text = ''
                for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            raw_text += text
                content = raw_text 
            folder_path = "docs"  # Path to the existing "docs" folder
            file_name = "outputA"
# Combine the folder path and file name to get the file path
            file_path = os.path.join(folder_path, file_name +".txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
                return content
#function to read texts uploaded  
        if text:
            folder_path = "docs"  # Path to the existing "docs" folder
            file_name = "outputA"
# Combine the folder path and file name to get the file path
            file_path = os.path.join(folder_path, file_name +".txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
                return text
            
         
     
    def user(user_message, history):
#convert the content of the outputA.txt to embeddings
                loader = TextLoader("docs\outputA.txt")
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000,separator='\n', chunk_overlap=0)
                documents = text_splitter.split_documents(documents)
                docsearch = FAISS.from_documents(documents, embeddings)
#question and answering with conversationalRetrievalChain
                qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), docsearch.as_retriever(), return_source_documents=True)
                print("user message:", user_message)
                print("Chat history", history)

                chat_history_str = "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])


#Get response from QA chain
                response=qa({"question":user_message, "chat_history":chat_history_str})
#Append user messgae and response to chat history
                history.append((user_message, response["answer"]))
                print ("Updated chat history:", history )
                return gr.update(value=""), history


#Gradio Interface
    gr.themes.colors.fuchsia
    gr.Markdown("""
    # SOFT WORK!
    Navigate through the tabs to transcribe your Audio/Video files via the AUDIO/VIDEO Tab,
    output the content of your PDF file via the PDF FILE tab
    and interact with any of the files via the CHATBOT TAB.
    """)
    with gr.Tab("Audio/Video"):
            gr.themes.colors.amber
            # audio_input = gr.inputs.Audio(source="upload", type="filepath")
            with gr.Row():
                vid=gr.Video(source="upload",format=None)
                aud=gr.inputs.Audio(source="upload", type="filepath")
            with gr.Row():
                txt=gr.Textbox(label="Upload a link to a youtube video")
                
            transcribe_button = gr.Button("Transcribe")

            transcribe_button.click(fn=transcribe,
                                    inputs=[vid,aud,txt],
            outputs= gr.outputs.Textbox(label="Transcription"))
            
    with gr.Tab("PDF FILE/TEXT"):
            with gr.Row():
                 file=gr.inputs.File(label="Upload a File")
                 text=gr.Textbox(label="Upload a text",interactive=True)
            submit_button=gr.Button("Submit")
            submit_button.click(fn=ret,
                                    inputs=[file,text],
            outputs= gr.outputs.Textbox(label="Content"))


    with gr.Tab("Chatbot"):
            chatbot=gr.Chatbot([], elem_id="chatbot",label='Chatbot').style(height=450)
            with gr.Row():
                #with gr.Column(scale=0.80):
                msg=gr.Textbox(placeholder="ask a question from your audio/video or PDF file",interactive=True).style(container=False)
            with gr.Row():
                with gr.Column(scale=0.50):
                    submit = gr.Button(
                        'Submit',
                        variant='primary'
                    )
                with gr.Column(scale=0.50):
                    clear = gr.Button(
                        'Clear',
                        variant='stop'
                    )
                    chat_history=[]
    submit.click(user,[msg,chatbot],[msg,chatbot], queue=False)
    msg.submit(user,[msg,chatbot],[msg,chatbot], queue=False)
    clear.click(lambda:None, None, chatbot, queue = False)
demo.launch(share=True)




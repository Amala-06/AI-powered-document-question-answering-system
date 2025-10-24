import gradio as gr
from transformers import pipeline
import PyPDF2

qa_object =pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

def extractPDFText(pdffile):
    text=""
    readPDF=PyPDF2.PdfReader(pdffile)
    
    for pg in readPDF.pages:
        txt=pg.extract_text()
        if txt:
            text += txt +"\n"
            
    return text

def answer_question(pdf,question):
    
    if pdf is None or len(question.strip())==0:
        return "Please provide both the PDF and the question."
        
    document = extractPDFText(pdf)

    if len(document.strip())==0:
        return "The provided PDF was empty."

    op=qa_object(question=question, context=document)

    return op['answer']

iface= gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(file_types=[".pdf"],label="Upload PDF File"),
        gr.Textbox(lines=2,placeholder="Enter your question")
    ],
    outputs="text",
    title="AI powered document question-answering system",
    description="Upload a PDF and ask a question. The AI answers based on the PDF content."
    
)
    
iface.launch() 

    
  
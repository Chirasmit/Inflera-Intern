# Inflera-Intern
Assignment: Build a RAG-Powered Multi-Agent Q&A Assistant

So I have attached two files in the git hub repo.
1)Lansera
2)check2.py

"Lasera is the main project file"

Lansera is a document analysis and question-answering tool built with Streamlit, Cohere, TensorFlow, and other useful libraries. This tool allows users to upload documents in various formats (TXT, PDF, PPTX, DOCX) and then ask questions related to the content of the uploaded documents. It also supports image prediction for images extracted from these documents.
Features
Document Upload and Text Extraction: Upload documents in TXT, PDF, PPTX, and DOCX formats, and the tool extracts the text content.
Question Answering: Users can input questions, and the tool uses Cohere's AI models to generate contextual answers based on the document's content.
Image Prediction: Images embedded in PPTX, DOCX, and PDF files are extracted, and predictions are made using a pre-trained MobileNetV2 model.
Real-Time Processing: The entire process, from file upload to question answering, is done in real-time using Streamlit’s interactive interface.
Technologies Used
Streamlit: For creating the interactive web app.
Cohere: For providing language models to answer user queries about the document.
TensorFlow: For image processing and prediction using MobileNetV2.
PyMuPDF (fitz): For extracting text and images from PDFs.
python-pptx: For extracting text and images from PPTX files.
python-docx: For extracting text and images from DOCX files.
OpenCV: For image processing.
Getting Started
Prerequisites
Before running the app, ensure you have the following installed:

Python 3.6+
Streamlit
Cohere API Key
TensorFlow
OpenCV
pytesseract
PyMuPDF (fitz)
python-pptx
python-docx
You can install the necessary Python libraries with the following command:


pip install streamlit cohere tensorflow opencv-python pytesseract PyMuPDF python-pptx python-docx

Now all the things required are in the file named Lansera excpet point number 2
Vector Store & Retrieval

when i wrote and added the point 2 in the code then it came in my ui that 

"RuntimeError: Failed to import transformers.integrations.integration_utils because of the following error (look up to see its traceback): Failed to import transformers.modeling_tf_utils because of the following error (look up to see its traceback): Another metric with the same name already exists."

which didn allowed me to add the 2nd point but the project lansera is also way more efficient in doing the required task.

Thanks for reading the content.I hope I made u understand it.





from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadPDFform
from .models import PDF

import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .config import OPENAI_KEY, PINCONE_KEY, ENV

def split_docs(doc, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    broken = text_splitter.split_documents(doc)
    return broken

directory = 'files/pdfs'

def load_docs(directory):
   loader = DirectoryLoader(directory)
   documents = loader.load()
   return documents

documents = load_docs(directory)

split = split_docs(documents)

# print(split[0].page_content)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

pinecone.init(
    api_key=PINCONE_KEY,
    environment=ENV,
    )

index_name = "langchain-demo"

if index_name not in pinecone.list_indexes():
   pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536
   )

ind = Pinecone.from_documents(split, embeddings, index_name=index_name)

def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = ind.similarity_search_with_score(query, k=k)
  else:
    similar_docs = ind.similarity_search(query, k=k)
  return similar_docs

model_name = "gpt-3.5-turbo"
llm = OpenAI(model_name=model_name, openai_api_key=OPENAI_KEY, temperature=0)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

query = "What is Multilayer perceptron?"
ans = get_answer(query)
print(ans)

# Create your views here.

def upload_pdf(request):
    if request.method == 'POST':
        form = UploadPDFform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse("The file is uploaded!")
    else:
        form = UploadPDFform()
        context = {
            'form': form,
        }
    return render(request, 'upload_pdf.html', context)

def index(request): 
    docs = PDF.objects.all()
    x = ""
    if request.method == 'POST':
        file = request.POST.get('file')
        path1 = os.path.join(BASE_DIR, "files")
        path2 = os.path.join(path1, file)
        try:
            PDF.objects.filter(pdf=file).delete()
            os.remove(path2)
        except Exception as e:
            x = str(e)
        return render(request, 'index.html', {'pdfs' : docs, 'x' : x})
    return render(request, 'index.html', {'pdfs' : docs})





# def qa(request):
#     return

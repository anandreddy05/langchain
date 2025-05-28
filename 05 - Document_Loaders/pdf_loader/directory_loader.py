# TO load all the pdfs,csvs etc from a directory/folder
# Use Lazy Load when we have more pdfs
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path="",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
# docs = loader.load()
docs = loader.lazy_load() # -> lazy loader makes a generator which loads on demand.


print(len(docs))
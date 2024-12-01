from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
import streamlit as st
import dotenv
import re

from utils import clone_repository, get_main_files_content, get_huggingface_embeddings, extract_repo_name

# Load environment variables
dotenv.load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("codebase-rag")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.gentenv("GROQ_API_KEY")
)


def get_indexed_repositories():
    """Get list of repositories that have been indexed in Pinecone"""
    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.namespaces.keys()
        return set(namespaces)
    except Exception as e:
        st.error(f"Error fetching repositories: {str(e)}")
        return set()

def remove_repository(repo_name):
    """Remove a repository from Pinecone"""
    try:
        pinecone_index.delete(namespace=repo_name, delete_all=True)
        return True
    except Exception as e:
        st.error(f"Error removing repository: {str(e)}")
        return False

def initialize_vectorstore(repo_url):
    """Initialize the vector store with documents from the specified repository"""
    repo_name = extract_repo_name(repo_url)
    if not repo_name:
        raise ValueError("Invalid GitHub repository URL")

    repo_path = clone_repository(repo_url)
    file_content = get_main_files_content(repo_path)
    documents = []
    
    for file in file_content:
        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={
                "source": file['name'],
                "repo": repo_name,
                "text": f"{file['name']}\n{file['content']}"
            }
        )
        documents.append(doc)

    return PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(),
        index_name="codebase-rag",
        namespace=repo_name
    )

def perform_rag(query, selected_repos):
    """Perform RAG across multiple repositories and return responses and sources"""
    raw_query_embedding = get_huggingface_embeddings(query)
    
    all_contexts = []
    all_sources = []
    
    for repo in selected_repos:
        top_matches = pinecone_index.query(
            vector=raw_query_embedding.tolist(), 
            top_k=3,  # Reduced to accommodate multiple repos
            include_metadata=True, 
            namespace=repo
        )

        for item in top_matches['matches']:
            all_contexts.append(item['metadata']['text'])
            if 'source' in item['metadata']:
                all_sources.append(f"{repo}/{item['metadata']['source']}")

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(all_contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are a Senior Software Engineer, with deep knowledge of multiple programming languages.
    Answer any questions I have about the codebases, think step by step and based on the code provided. Always consider all of the context provided when forming a response.
    When referencing code, specify which repository it comes from.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content, list(set(all_sources))




############################################### Streamlit UI ###############################################
st.set_page_config(page_title="Talk to your Codebase", page_icon="💻", layout="wide")
st.title("Talk to your Codebase")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_repos" not in st.session_state:
    st.session_state.selected_repos = []

# Load existing repositories from Pinecone
st.session_state.repositories = get_indexed_repositories()

# Sidebar for repository management
with st.sidebar:
    st.header("Repository Management")
    
    # Add new repository
    new_repo = st.text_input("Add GitHub Repository URL", 
                            placeholder="https://github.com/username/repo")
    if st.button("Add Repository"):
        if new_repo:
            repo_name = extract_repo_name(new_repo)
            if repo_name and repo_name not in st.session_state.repositories:
                with st.spinner(f"Indexing repository {repo_name}..."):
                    try:
                        initialize_vectorstore(new_repo)
                        st.session_state.repositories.add(repo_name)
                        st.success(f"Successfully added {repo_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding repository: {str(e)}")
            else:
                st.warning("Invalid repository URL or repository already added")
    
    # Repository selector (multiple)
    if st.session_state.repositories:
        st.session_state.selected_repos = st.multiselect(
            "Select Repositories",
            options=sorted(list(st.session_state.repositories)),
            default=st.session_state.selected_repos if st.session_state.selected_repos else None
        )
        
        # Remove repository section
        st.subheader("Remove Repository")
        repo_to_remove = st.selectbox(
            "Select repository to remove",
            options=sorted(list(st.session_state.repositories)),
            key="repo_to_remove"
        )
        if st.button("Remove Selected Repository"):
            if repo_to_remove:
                if remove_repository(repo_to_remove):
                    st.session_state.repositories.remove(repo_to_remove)
                    if repo_to_remove in st.session_state.selected_repos:
                        st.session_state.selected_repos.remove(repo_to_remove)
                    st.success(f"Successfully removed {repo_to_remove}")
                    st.rerun()
    
    # Show current repositories
    if st.session_state.repositories:
        st.subheader("Available Repositories:")
        for repo in sorted(st.session_state.repositories):
            st.write(f"• {repo}")
    else:
        st.info("No repositories added yet. Add a repository to start chatting!")

# Main chat interface
if st.session_state.selected_repos:
    st.markdown("""
    Ask questions about the selected repositories' codebases!
    The assistant will reference relevant files to provide accurate information.
    """)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.markdown("**Referenced Files:**")
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

    # Chat input
    if prompt := st.chat_input("Ask about the codebase..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "repos": st.session_state.selected_repos.copy()
        })
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response, sources = perform_rag(prompt, st.session_state.selected_repos)
            st.markdown(response)
            if sources:
                st.markdown("**Referenced Files:**")
                for source in sources:
                    st.markdown(f"- `{source}`")
            
            # Add response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "repos": st.session_state.selected_repos.copy()
            })
else:
    st.info("Please select one or more repositories from the sidebar to start chatting!")
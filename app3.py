import validators, streamlit as st
import traceback
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and URL
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## LLM Setup using Groq API
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_video_id(url):
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query).get("v", [None])[0]

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website)")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                # Load YouTube transcript or website content
                if "youtube.com" in generic_url:
                    video_id = extract_video_id(generic_url)
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    full_text = " ".join([entry["text"] for entry in transcript])
                    docs = [Document(page_content=full_text)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )
                    docs = loader.load()

                # Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("✅ Summary Generated:")
                st.write(output_summary)

        except Exception as e:
            st.error("An error occurred:")
            st.code(traceback.format_exc())

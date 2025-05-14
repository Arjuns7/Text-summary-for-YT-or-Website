import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="🦜")
st.title("🦜 Langchain : Summarize Text from Yt or Website")
st.subheader('Summarize URL')


with st.sidebar:
    groq_api_key = st.text_input("Enter groq api key",type="password")

genric_url = st.text_input("URL", label_visibility="collapsed")
llm = ChatGroq(model="Gemma-7b-It",groq_api_key=groq_api_key)

prompt_template ='''
Provide the summary of following content in 300 words:
content:{text}
'''
prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not genric_url.strip():
        st.error("Please provide the info")
    elif not validators.url(genric_url):
        st.error("Please enter a valid url")

    else:
        try:
            with st.spinner("Waiting...."):
                if "youtube.com" in genric_url:
                    loader = YoutubeLoader.from_youtube_url(genric_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[genric_url],ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})


                data=loader.load()

                chain = load_summarize_chain(llm,chain_type='stuff',prompt=prompt)
                output = chain.run(data)

                st.success(output)

        except Exception as e:
            st.exception(f"Exception :{e}")
    
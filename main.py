from langdetect import detect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# Config
# -------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

instructions = {
    "en": "Answer ONLY using the medical context below. If the answer is not present, reply 'I don’t know'.",
    "or": "କୃପୟା ନିମ୍ନୋଲିଖିତ ପ୍ରସଙ୍ଗରୁ ମାତ୍ର ଉତ୍ତର ଦିଅନ୍ତୁ | ଉତ୍ତର ନଥିଲେ 'ମୁଁ ଜାଣିନି' କୁହନ୍ତୁ |",  # Odia
    "mr": "फक्त खालील संदर्भ वापरून उत्तर द्या. उत्तर उपलब्ध नसेल तर 'मला माहित नाही' असे म्हणा.",  # Marathi
    "ur": "صرف نیچے دیے گئے سیاق و سباق کا استعمال کرتے ہوئے جواب دیں۔ اگر جواب موجود نہیں ہے تو 'مجھے نہیں معلوم' کہیں۔",  # Urdu
    "ta": "கீழே உள்ள சூழலை மட்டுமே பயன்படுத்தி பதிலளிக்கவும். பதில் இல்லாவிட்டால் 'எனக்கு தெரியவில்லை' என்று சொல்லவும்.",  # Tamil
    "te": "క్రింద ఇచ్చిన సందర్భం ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి. సమాధానం లేకపోతే 'నాకు తెలియదు' అని చెప్పండి.",  # Telugu
}

# -------------------------
# Build RAG pipeline
# -------------------------
def load_vectorstore():
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vect = FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)
    return vect

def build_prompt(query, lang):
    instruction_text = instructions.get(lang, instructions["en"])
    return f"""
You are a multilingual medical assistant.
{instruction_text}
Always answer in the SAME language as the question.

Context:
{{context}}

Question: {query}

Answer:
"""

def build_chain(vect, query, lang):
    llm = Ollama(model="phi3:mini")  # or smaller llama3 variant if GPU is limited
    retriever = vect.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt = PromptTemplate.from_template(build_prompt(query, lang))
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

# -------------------------
# Interactive Loop
# -------------------------
if __name__ == "__main__":
    vect = load_vectorstore()
    print("✅ Chatbot ready. Type 'exit' to quit.")

    while True:
        query = input("\n👤 You: ")
        if query.lower() in ["exit", "quit"]:
            break

        try:
            lang = detect(query)
        except:
            lang = "en"

        qa = build_chain(vect, query, lang)
        result = qa.invoke({"query": query})

        print("\n🤖 Bot:", result["result"])
        print("📎 Sources:", [d.metadata for d in result["source_documents"]])

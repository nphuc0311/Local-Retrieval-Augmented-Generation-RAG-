from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

class Generate:
    def __init__(self, retriever, router):
        self.retriever = retriever
        self.router = router
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
        self.memory = ConversationBufferMemory(chat_memory_key="chat_history", return_messages=True)

    def generate_answer(self, question: str, top_k: int) -> str:
        query_type = self.router.route(question)

        history = self.memory.load_memory_variables({}).get("history", [])

        if query_type == "rag_query":
            context = str(self.retriever.search(question, top_k))
            template = """Bạn là trợ lý ảo thân thiện của Kicap - cửa hàng bàn phím cơ.
            Sử dụng thông tin sau để trả lời câu hỏi một cách tự nhiên.
            Nếu không biết câu trả lời, hãy nói không có thông tin sản phẩm.

            Lịch sử hội thoại trước đó:
            {history}

            Context:
            {context}

            Câu hỏi: {question}

            Trả lời bằng tiếng Việt:"""

        else:
            context = []
            template = """Bạn là trợ lý ảo thân thiện của Kicap - cửa hàng bàn phím cơ.
            Hãy trả lời câu hỏi một cách tự nhiên bằng tiếng Việt.

            Lịch sử hội thoại trước đó:
            {history}

            Câu hỏi: {question}

            Trả lời:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            RunnablePassthrough.assign(history=lambda _: history)  # Lấy lịch sử hội thoại
            | RunnablePassthrough.assign(context=lambda x: x["context"])  # Gán context vào
            | prompt
            | self.llm
        )

        raw_answer = chain.invoke({"context": context, "question": question})

        if hasattr(raw_answer, "content"):
            answer = raw_answer.content
            answer = str(raw_answer)

        self.memory.save_context({"question": question}, {"answer": answer})

        return answer, context

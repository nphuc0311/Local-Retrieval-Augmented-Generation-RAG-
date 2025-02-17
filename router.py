from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class Router:
  def __init__(self):
    self.router_prompt = ChatPromptTemplate.from_messages([
      ("system", """Bạn là chuyên gia phân loại câu hỏi cho cửa hàng bàn phím cơ Kicap. Phân loại thành:
  - 'rag_query' nếu câu hỏi liên quan đến việc gợi ý sản phẩm bàn phím cơ, đặc tính kỹ thuật, so sánh sản phẩm hoặc yêu cầu thông tin chi tiết về:
    + Thông số: switch (Yunmu, Azure, Outemu...), loại kết nối, dung lượng pin
    + Vật liệu: khung nhôm CNC, PBT doubleshot...
    + Tính năng: hotswap, gasket mount, RGB
    + So sánh giữa các model/bản build
    + Tìm sản phẩm theo tiêu chí cụ thể
  - 'general_query' cho các câu hỏi chung về:
    + Hướng dẫn sử dụng cơ bản
    + Chính sách mua hàng/đổi trả
    + Câu hỏi không xác định (ví dụ: chào hỏi, hỏi random)
    + Yêu cầu không liên quan đến sản phẩm

  Ví dụ:
  Q: Tôi muốn mua bàn phím cơ, hãy gợi ý cho tôi một sản phẩm phù hợp.
  A: rag_query

  Q: Cách đổi keycap?
  A: general_query"""),

      ("human", "Câu hỏi cần phân loại: {question}"),

      ("human", """Chỉ trả lời 1 từ (rag_query/general_query).
  Các từ khoá quan trọng cần xem xét:
  [switch, layout, hotswap, foam, pin, CNC, doubleshot, gasket, RGB, kit, mods, wireless, bàn phím cơ]""")
  ])

    self.chain = (self.router_prompt | ChatOpenAI(temperature=0, model="gpt-4o-mini") | StrOutputParser())

  def route(self, question: str) -> str:
    response = self.chain.invoke({"question": question})
    return response.strip().lower()
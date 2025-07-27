import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import docx

MODEL_NAME = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text
docx_file_path = r"content/class 7 Social chapter-1.docx"

context = extract_text_from_docx(docx_file_path)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )
    return answer if answer.strip() else "Sorry, I couldn't find an answer."
print("✅ bertmodel.py loaded and context ready.")
print("Chatbot: Ask me anything about Class 7 Social Science Chapter-1 (type 'exit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = answer_question(user_input, context)
    print("Chatbot:", response)

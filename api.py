from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import PeftModel
import torch

app = FastAPI()

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(model, "saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")
model.eval()

class QARequest(BaseModel):
    question: str
    context: str

@app.post("/predict")
def predict(req: QARequest):
    inputs = tokenizer(
        req.question, req.context,
        max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    answer = tokenizer.decode(inputs["input_ids"][0][start:end+1])
    return {"answer": answer}

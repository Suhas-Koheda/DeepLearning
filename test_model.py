import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

MODEL_PATH = "/home/ssp/model/codet5-fast"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_PREFIX = "Optimize Java: "

def test_model():
    print(f"Loading model from {MODEL_PATH}")
    print(f"Using device: {DEVICE}")
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Test cases
    test_cases = [
        """String alertLevel;
switch (sensor.getValue()) {
    case 0:
    case 1:
        alertLevel = "GREEN";
        break;
    case 2:
        alertLevel = "YELLOW";
        break;
    case 3:
        alertLevel = "RED";
        break;
    default:
        alertLevel = "CRITICAL";
}""",
        
        """List<Payment> payments = new ArrayList<>(pendingPayments);
Iterator<Payment> iterator = payments.iterator();
while (iterator.hasNext()) {
    Payment p = iterator.next();
    if (p.isExpired()) {
        iterator.remove();
    }
}""",
        
        """String result = "";
for (int i = 0; i < 100; i++) {
    result += "Value: " + i + "\\n";
}"""
    ]
    
    for i, code in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print("Input:")
        print(code)
        
        inputs = tokenizer(
            TASK_PREFIX + code.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )
        
        optimized_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Output:")
        print(optimized_code.strip())
        print("-" * 50)

if __name__ == "__main__":
    test_model()
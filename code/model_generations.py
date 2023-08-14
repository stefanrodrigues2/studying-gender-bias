from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

def generate_output_toxicity(prompts, model_name):
    model_continuations = []
    text_generation = pipeline("text-generation", model=model_name)
    for prompt in prompts:
        generation = text_generation(prompt, max_length=50, do_sample=True, pad_token_id=50256, top_p=1.0, top_k=50)
        continuation = generation[0]['generated_text'].replace(prompt,'')
        model_continuations.append(continuation)
    return model_continuations

def generate_output_regard(prompt, text_generation):
    model_continuations = []
    generation = text_generation(prompt, max_length=50, do_sample=True, pad_token_id=50256, top_p=1.0, top_k=50)
    continuation = generation[0]['generated_text'].replace(prompt,'')
    model_continuations.append(continuation)
    return model_continuations

def generate_toxicity_t5(prompts, model_name):
    model_continuations = []
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=1.5, top_k=50,repetition_penalty=2.0, top_p=0.95)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        model_continuations.append(output_text)
    return model_continuations
    
def generate_regard_t5(prompt,model,tokenizer):
    model_continuations = []
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=1.5, top_k=50,repetition_penalty=2.0, top_p=0.95)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model_continuations.append(output_text)
    return model_continuations

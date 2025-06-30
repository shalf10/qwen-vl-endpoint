from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import base64
import io
import torch

model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        model = carregar_modelo()
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # mais seguro que float16
        device_map="auto"
    )
    print("✅ Modelo carregado com sucesso.")

def decode_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print("❌ Erro ao converter imagem:", e)
        return None

def handler(event):
    try:
        global model, processor
        if model is None or processor is None:
            load_model()

        prompt = event["input"].get("prompt", "")
        image_base64 = event["input"].get("image_base64", "")

        if not prompt or not image_base64:
            return {"status": "erro", "mensagem": "Prompt ou imagem ausente."}

        image = decode_image(image_base64)
        if image is None:
            return {"status": "erro", "mensagem": "Erro ao decodificar imagem."}

        inputs = processor(prompt=prompt, images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512)
        decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

        return {
            "status": "ok",
            "resposta": decoded,
            "prompt_usado": prompt
        }

    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

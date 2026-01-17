# 🌙 Saad Saudi Arabic Model | نموذج سعد

**Training code for Saudi Arabic conversational AI model**

**كود تدريب نموذج محادثة سعودي**

---

## 👨‍💻 Developer | المطور

**Abdullah Al-Shareef | عبدالله الشريف**

Saudi developer specialized in AI and NLP

---

## ✨ Features | المميزات

- 🇸🇦 Saudi dialect understanding | فهم اللهجة السعودية
- 💬 Modern Standard Arabic support | دعم العربية الفصحى
- 🤖 Natural conversation | محادثة طبيعية
- 💎 LoRA + 4-bit quantization | تدريب فعال

---

## 🛠️ Installation | التثبيت

```bash
git clone https://github.com/iamdev7/saad-saudi-model-training.git
cd saad-saudi-model-training
pip install -r requirements.txt
```

---

## 📊 Quick Start | البدء السريع

### 1️⃣ Prepare Data | تحضير البيانات

```bash
# Place your data files in data/raw/
# ضع ملفات البيانات في data/raw/

python prepare_data.py
```

**Supported formats | التنسيقات المدعومة:**
- JSON: `[{"user": "...", "assistant": "..."}]`
- TXT: `User: ... | Assistant: ...`

### 2️⃣ Train | التدريب

```bash
python train_model.py
```

**Training features | مميزات التدريب:**
- LoRA fine-tuning (~1-2% of parameters)
- 4-bit quantization for efficiency
- Automatic checkpointing
- GPU + CPU support

### 3️⃣ Upload | الرفع

```bash
python upload_to_hf.py
```

---

## 💻 Usage Example | مثال الاستخدام

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("A7be7/saad")
tokenizer = AutoTokenizer.from_pretrained("A7be7/saad")

# Generate
prompt = "مرحبا، كيف حالك؟"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0])
print(response)
```

---

## 📂 Project Structure | هيكل المشروع

```
├── data/
│   ├── raw/              # Your data here
│   └── processed/       # Processed data
├── models/
│   └── saad-saudi-model/  # Trained model
├── prepare_data.py
├── train_model.py
├── upload_to_hf.py
└── requirements.txt
```

---

## ⚙️ Configuration | الإعدادات

Edit `train_model.py` for custom settings:

```python
base_model = "aubmindlab/aragpt2-base"
num_epochs = 3
batch_size = 4
learning_rate = 2e-4
max_length = 512
```

---

## 💾 Requirements | المتطلبات

- Python 3.8+
- CUDA 11.8+ (recommended)
- 16GB RAM
- GPU with 8GB+ VRAM (12GB recommended)

---

## 📝 License | الترخيص

CreativeML OpenRAIL-M

---

## 🔗 Links | الروابط

- **Model on Hugging Face**: [A7be7/saad](https://huggingface.co/A7be7/saad)
- **GitHub**: [iamdev7/saad-saudi-model-training](https://github.com/iamdev7/saad-saudi-model-training)

---

<div align="center">

**Made with ❤️ in Saudi Arabia 🇸🇦**

**صُنع بكل حب في السعودية**

</div>

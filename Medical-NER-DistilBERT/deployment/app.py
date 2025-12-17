import gradio as gr
from transformers import pipeline
import re

# --- 1. SETUP MODEL ---
# Load model dari folder lokal (yang sudah diupload)
model_path = "." 

print("⏳ Sedang memuat model AI...")
try:
    # Aggregation strategy "simple" penting agar outputnya rapi
    ner_pipeline = pipeline(
        "token-classification", 
        model=model_path, 
        tokenizer=model_path, 
        aggregation_strategy="simple"
    )
    print("✅ Model AI berhasil dimuat!")
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    ner_pipeline = None

# --- 2. FUNGSI LOGIKA HYBRID (AI + REGEX) ---
def analisis_medis_hybrid(teks):
    # A. Jalankan AI (Untuk Obat & Penyakit)
    entities = []
    if ner_pipeline:
        entities = ner_pipeline(teks)
    
    # B. Jalankan Regex (Untuk Dosis)
    # Pola: Angka + (Spasi Opsional) + Satuan (mg, ml, tablet, dll)
    pola_dosis = r"(\d+(\.\d+)?\s?(mg|g|ml|mcg|L|oz|tablet|tabs|capsule|cap|pills|daily|x daily))"
    
    for match in re.finditer(pola_dosis, teks, re.IGNORECASE):
        # Kita format hasil Regex agar strukturnya sama dengan hasil AI
        dosis_entity = {
            "entity_group": "DOSAGE",     # Label manual
            "score": 1.0,                 # Kita yakin 100% karena ini logika pasti
            "word": match.group(),
            "start": match.start(),
            "end": match.end()
        }
        entities.append(dosis_entity)

    # C. Urutkan semua entitas berdasarkan posisi munculnya di teks
    entities.sort(key=lambda x: x['start'])

    # D. Format Output untuk Gradio (Potong-potong teks)
    formatted_results = []
    cursor = 0
    
    for ent in entities:
        start, end = ent['start'], ent['end']
        label = ent['entity_group']
        
        # Cek overlap (Tumpang tindih)
        # Jika Regex mendeteksi sesuatu di dalam area yang sudah ditandai AI, lewati regexnya
        if start < cursor:
            continue

        # Masukkan teks biasa sebelum entitas
        if start > cursor:
            formatted_results.append((teks[cursor:start], None))
        
        # Masukkan entitas berwarna
        formatted_results.append((teks[start:end], label))
        cursor = end
    
    # Masukkan sisa teks di akhir kalimat
    if cursor < len(teks):
        formatted_results.append((teks[cursor:], None))
        
    return formatted_results

# --- 3. TAMPILAN WEB (FRONTEND) ---
custom_css = """
.gradio-container {background-color: #f0f4f8}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 Intelligent Medical Entity Recognition
        **Model:** Hybrid DistilBERT + Pattern Matching
        
        Sistem ini mendeteksi:
        * 🟦 **Chemical** (Obat-obatan) - *via AI*
        * 🟥 **Disease** (Penyakit) - *via AI*
        * 🟩 **Dosage** (Dosis & Satuan) - *via Regex Logic*
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Catatan Medis (Inggris)", 
                placeholder="Paste clinical text here...", 
                lines=5,
                value="The patient was prescribed Aspirin 500mg and Metformin 850 mg for type 2 diabetes."
            )
            btn = gr.Button("🔍 Analisis Lengkap", variant="primary")
            
            gr.Examples([
                ["Patient prescribed 500mg Amoxicillin 3x daily for bacterial infection."],
                ["Inject 10ml of Insulin for diabetes."],
                ["Diagnosis: Severe hypertension. Plan: Lisinopril 10 mg daily."]
            ], inputs=input_text)
        
        with gr.Column():
            output_display = gr.HighlightedText(
                label="Hasil Deteksi",
                combine_adjacent=True,
                show_legend=True,
                color_map={
                    "Chemical": "blue", 
                    "Disease": "red", 
                    "DOSAGE": "green" # Warna baru untuk Dosis!
                }
            )

    btn.click(fn=analisis_medis_hybrid, inputs=input_text, outputs=output_display)

# Jalankan App
if __name__ == "__main__":
    demo.launch()
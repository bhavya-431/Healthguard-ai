import torch
import os
import re
from src.gnn_model import MedicalGNN
from src.data_preprocessing import load_and_process_data, clean_text
import torch.nn.functional as F

class MedicalChatbot:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(current_dir, '..', 'dataset.csv')
        self.model_path = os.path.join(current_dir, '..', 'best_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Initializing Chatbot...")
        # Load full graph data (vocab is built once, deterministically)
        self.data_meta = load_and_process_data(self.csv_path)
        self.vocab = self.data_meta.vocab
        self.labels = {v: k for k, v in self.data_meta.label_mapping.items()}
        self.num_nodes = self.data_meta.num_nodes
        self.num_classes = self.data_meta.num_classes
        self.num_patients = self.data_meta.num_patients
        
        # Load Model
        self.model = MedicalGNN(self.num_nodes, 64, self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        # -------------------------------------------------------
        # KEY FIX: Pre-compute predictions for ALL patient nodes
        # using the full trained graph (exactly as done in training).
        # This is stored once at startup and reused at query time.
        # -------------------------------------------------------
        print("Pre-computing full-graph predictions for all patients...")
        data_on_device = self.data_meta.to(self.device)
        with torch.no_grad():
            out = self.model(data_on_device)            # [total_nodes, num_classes]
            # Only patient nodes (0 .. num_patients-1)
            self.patient_logits = out[:self.num_patients]   # [num_patients, num_classes]
        
        # Store per-patient word sets for similarity matching
        import pandas as pd
        import re as _re
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                          'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                          'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                          'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                          'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
                          'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                          'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                          'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                          'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                          'through', 'during', 'before', 'after', 'above', 'below', 'to',
                          'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                          'again', 'further', 'then', 'once', 'here', 'there', 'when',
                          'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                          'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                          'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                          'can', 'will', 'just', 'don', 'should', 'now'])

        df = pd.read_csv(self.csv_path)
        self.patient_word_sets = []
        for text in df['text']:
            clean = clean_text(str(text))
            words = set(w for w in clean.split() if w in self.vocab and w not in stop_words and len(w) > 2)
            self.patient_word_sets.append(words)

        print(f"Ready. {self.num_patients} patients loaded, {self.num_classes} disease classes.")

    def predict(self, text):
        """
        Prediction strategy:
        1. Clean and tokenize the user input.
        2. Find the top-K most similar patients by Jaccard word overlap.
        3. Average their GNN logits (weighted by similarity).
        4. Return top-3 predictions from the averaged logits.
        """
        cleaned = clean_text(text)
        query_words = set(w for w in cleaned.split() if w in self.vocab and len(w) > 2)

        if not query_words:
            return [{
                "disease": "Unknown",
                "confidence": 0.0,
                "precautions": ["Please describe your symptoms in more detail."]
            }]

        # Jaccard similarity against every patient
        K = 20  # blend top-K patients
        similarities = []
        for i, patient_words in enumerate(self.patient_word_sets):
            if not patient_words:
                similarities.append(0.0)
                continue
            intersection = len(query_words & patient_words)
            union = len(query_words | patient_words)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

        # Get top-K
        import torch
        sim_tensor = torch.tensor(similarities, dtype=torch.float)
        topk_vals, topk_idx = sim_tensor.topk(min(K, self.num_patients))

        # Only use patients with non-zero similarity
        valid_mask = topk_vals > 0
        if valid_mask.sum() == 0:
            # No word overlap – fall back to full average
            avg_logits = self.patient_logits.mean(dim=0)
        else:
            valid_idx = topk_idx[valid_mask]
            valid_weights = topk_vals[valid_mask]
            valid_weights = valid_weights / valid_weights.sum()  # normalise

            # Weighted average of logits for these patients
            selected_logits = self.patient_logits[valid_idx.to(self.device)]  # [K', C]
            valid_weights = valid_weights.to(self.device).unsqueeze(1)         # [K', 1]
            avg_logits = (selected_logits * valid_weights).sum(dim=0)         # [C]

        probs = F.softmax(avg_logits, dim=0)

        # Top-3 predictions
        k_top = min(3, self.num_classes)
        top_probs, top_indices = probs.topk(k_top)

        results = []
        for i in range(k_top):
            disease = self.labels[top_indices[i].item()]
            conf = top_probs[i].item()
            precautions = self.get_precautions(disease)
            results.append({
                "disease": disease,
                "confidence": conf,
                "precautions": precautions
            })

        return results

    def get_precautions(self, disease):
        precautions_db = {
            'Psoriasis': ["Apply moisturizing creams", "Avoid triggers like stress and smoking", "Use medicated shampoos", "Expose skin to small amounts of sunlight"],
            'Varicose Veins': ["Elevate your legs", "Wear compression stockings", "Exercise regularly", "Avoid standing for long periods"],
            'Typhoid': ["Drink plenty of fluids", "Eat light, easy-to-digest foods", "Maintain good hygiene", "Complete the full course of antibiotics"],
            'Chicken pox': ["Avoid scratching", "Take oatmeal baths", "Wear loose clothing", "Isolate to prevent spreading"],
            'Impetigo': ["Keep the infected area clean", "Use antibiotic ointment", "Wash clothes and towels daily", "Avoid touching the sores"],
            'Dengue': ["Drink plenty of fluids", "Take rest", "Use mosquito repellents", "Take acetaminophen for fever"],
            'Fungal infection': ["Keep the area dry and clean", "Use antifungal creams", "Wear loose-fitting clothes", "Avoid sharing personal items"],
            'Common Cold': ["Drink warm fluids", "Rest", "Gargle with salt water", "Steam inhalation"],
            'Pneumonia': ["Get plenty of rest", "Drink fluids", "Take prescribed antibiotics", "Use a humidifier"],
            'Dimorphic Hemorrhoids': ["Eat high-fiber foods", "Drink plenty of water", "Take warm sitz baths", "Avoid straining during bowel movements"],
            'Arthritis': ["Exercise regularly", "Maintain a healthy weight", "Use hot and cold therapy", "Protect your joints"],
            'Acne': ["Wash face regularly", "Avoid touching your face", "Use non-comedogenic products", "Limit sun exposure"],
            'Bronchial Asthma': ["Avoid triggers (dust, pollen)", "Use inhalers as prescribed", "Monitor breathing", "Stay away from smoke"],
            'Hypertension': ["Reduce salt intake", "Exercise regularly", "Manage stress", "Limit alcohol consumption"],
            'Migraine': ["Rest in a quiet, dark room", "Apply cold or hot compresses", "Hydrate", "Identify and avoid triggers"],
            'Cervical spondylosis': ["Exercise neck muscles", "Maintain good posture", "Use a firm pillow", "Apply heat or cold"],
            'Jaundice': ["Drink plenty of water", "Eat light foods", "Avoid alcohol", "Rest"],
            'Malaria': ["Use mosquito nets", "Take antimalarial medication", "Wear long sleeves", "Drain standing water"],
            'Hepatitis A': ["Practice good hygiene", "Avoid alcohol", "Rest", "Eat small, frequent meals"],
            'Hepatitis B': ["Avoid alcohol", "Eat a healthy diet", "Rest", "Talk to a doctor about antiviral meds"],
            'Hepatitis C': ["Avoid alcohol", "Eat a balanced diet", "Avoid certain medications", "Get vaccinated for Hep A and B"],
            'Hepatitis D': ["Consult a specialist", "Avoid alcohol", "Maintain a healthy lifestyle", "Monitor liver health"],
            'Hepatitis E': ["Rest", "Drink plenty of fluids", "Avoid alcohol", "Practice good hygiene"],
            'Alcoholic hepatitis': ["Stop drinking alcohol", "Eat a nutritional diet", "Taking vitamin supplements", "Rest"],
            'Tuberculosis': ["Take prescribed medication", "Cover mouth when coughing", "Wear a mask", "Open windows for ventilation"],
            'Heart attack': ["Call emergency services immediately", "Chew aspirin if advised", "Stay calm", "Perform CPR if unconscious"],
            'Hypothyroidism': ["Take thyroid medication", "Eat a balanced diet", "Exercise", "Manage stress"],
            'Hyperthyroidism': ["Take prescribed medication", "Eat a healthy diet", "Avoid iodine-rich foods", "Manage stress"],
            'Hypoglycemia': ["Eat fast-acting sugar", "Monitor blood sugar", "Eat regular meals", "Carry a snack"],
            'Osteoarthristis': ["Exercise regularly", "Maintain a healthy weight", "Use pain relief creams", "Physical therapy"],
            'Paroymsal  Positional Vertigo': ["Perform Epley maneuver", "Sleep with head elevated", "Sit up slowly", "Avoid sudden head movements"],
            'Urinary tract infection': ["Drink plenty of water", "Wipe from front to back", "Urinate after intercourse", "Avoid irritating products"],
            'Allergy': ["Avoid known allergens", "Take antihistamines", "Wear a medical alert bracelet", "Carry an epinephrine injector"],
            'GERD': ["Avoid spicy foods", "Eat smaller meals", "Don't lie down after eating", "Maintain a healthy weight"],
            'Chronic cholestasis': ["Eat a low-fat diet", "Drink plenty of water", "Take prescribed medication", "Avoid alcohol"],
            'Drug Reaction': ["Stop taking the drug", "Consult a doctor", "Drink water", "Apply cool compresses for rash"],
            'Peptic ulcer diseae': ["Avoid spicy and acidic foods", "Don't smoke", "Limit alcohol", "Manage stress"],
            'AIDS': ["Take ART medication", "Eat healthy", "Avoid infections", "Regular checkups"],
            'Diabetes ': ["Monitor blood sugar", "Eat a balanced diet", "Exercise", "Take medication"],
            'Gastroenteritis': ["Stay hydrated", "Rest", "Eat bland foods", "Wash hands frequently"]
        }
        return precautions_db.get(disease, ["Consult a doctor for specific advice", "Rest and stay hydrated", "Monitor symptoms", "Seek medical attention if symptoms worsen"])

if __name__ == "__main__":
    bot = MedicalChatbot()
    test_text = "I have a skin rash and joint pain with itching and scaly patches"
    results = bot.predict(test_text)
    for r in results:
        print(f"Prediction: {r['disease']} ({r['confidence']:.2%})")

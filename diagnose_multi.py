import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from inference import MedicalChatbot

# Representative samples from the dataset (one per disease label)
tests = [
    ("Psoriasis", "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."),
    ("Varicose Veins", "My calves have been cramping up when I walk or stand for long periods of time. There are bruise marks on my calves, which is making me worried. I feel tired very soon."),
    ("Typhoid", "I have constipation and belly pain, and it's been really uncomfortable. The belly pain has been getting worse and is starting to affect my daily life. Moreover, I get chills every night, followed by a mild fever."),
    ("Chicken pox", "I've been experiencing intense itching all over my skin, and it's driving me crazy. I also have a rash that's red and inflamed."),
    ("Dengue", "I am facing severe joint pain and vomitting. I have developed a skin rash that covers my entire body and is accompanied by intense itching."),
    ("Fungal infection", "All over my body, there has been a severe itching that has been followed by a red, bumpy rash. My skin also has a few darkened spots and a few tiny, nodular breakouts."),
    ("Common Cold", "I can't stop sneezing and my nose is really runny. I'm also really cold and tired all the time, and I've been coughing a lot."),
    ("Pneumonia", "I've been feeling really cold and tired lately, and I've been coughing a lot with chest pain. My heart is beating really fast too."),
    ("Arthritis", "My muscles have been feeling really weak, and my neck has been extremely tight. I've been experiencing a lot of stiffness when I walk about and my joints have been swollen."),
    ("Migraine", "I have been experiencing acidity and indigestion after meals, as well as frequent headaches and blurred vision."),
]

def run_tests():
    print("Loading model...")
    bot = MedicalChatbot()
    
    correct = 0
    total = len(tests)
    
    print(f"\n{'='*60}")
    print(f"  Running {total} accuracy tests")
    print(f"{'='*60}\n")
    
    for expected, text in tests:
        predictions = bot.predict(text)
        predicted_diseases = [p['disease'] for p in predictions]
        top = predictions[0]['disease']
        top_conf = predictions[0]['confidence']
        
        match = (top.lower() == expected.lower())
        if match:
            correct += 1
            
        status = "✓ CORRECT" if match else f"✗ WRONG (expected: {expected})"
        print(f"Input snippet: ...{text[:60]}...")
        print(f"  Top Prediction: {top} ({top_conf:.1%})")
        print(f"  All 3: {', '.join(predicted_diseases)}")
        print(f"  Status: {status}")
        print()
    
    print(f"{'='*60}")
    print(f"  Accuracy: {correct}/{total} ({correct/total:.0%})")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_tests()

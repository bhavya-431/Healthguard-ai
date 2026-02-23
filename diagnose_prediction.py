import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from inference import MedicalChatbot

def test_prediction():
    bot = MedicalChatbot()
    
    # Test case from dataset (Row 1)
    test_text = "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."
    expected_label = "Psoriasis"
    
    print(f"\nEvaluating Input: {test_text}")
    print(f"Expected Label: {expected_label}")
    
    predictions = bot.predict(test_text)
    
    print("\nModel Predictions:")
    for i, p in enumerate(predictions):
        print(f"{i+1}. {p['disease']} - Confidence: {p['confidence']:.2%}")
        
    if predictions[0]['disease'].lower() == expected_label.lower():
        print("\nSUCCESS: Match found!")
    else:
        print("\nFAILURE: Top prediction does not match expected label.")

if __name__ == "__main__":
    test_prediction()

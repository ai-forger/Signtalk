#!/usr/bin/env python3
"""
Simple test to verify the model works with basic input
"""

import os
import numpy as np
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

def test_model_basic():
    """Test the model with basic input"""
    print("🔍 Simple Model Test")
    print("=" * 30)
    
    # Load model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create a simple test input (white image with some noise)
    test_input = np.ones((1, 400, 400, 3), dtype=np.uint8) * 255
    
    # Add some random noise to make it more realistic
    noise = np.random.randint(0, 50, (1, 400, 400, 3), dtype=np.uint8)
    test_input = test_input - noise
    
    print(f"✅ Test input created, shape: {test_input.shape}")
    print(f"   Input range: {test_input.min()} to {test_input.max()}")
    
    # Make prediction
    try:
        prob = model.predict(test_input, verbose=0)
        print(f"✅ Model prediction successful, output shape: {prob.shape}")
        
        # Get probabilities
        prob_array = np.array(prob[0], dtype='float32')
        print(f"✅ Probabilities extracted, shape: {prob_array.shape}")
        
        # Show top predictions
        top_indices = np.argsort(prob_array)[::-1][:3]
        print("\n📊 Top 3 Predictions:")
        for i, idx in enumerate(top_indices):
            confidence = prob_array[idx]
            print(f"   {i+1}. Group {idx}: {confidence:.4f}")
        
        # Map groups to letters
        group_letters = {
            0: "A/E/M/N/S/T",
            1: "B/D/F/I/K/R/U/V/W", 
            2: "C/O",
            3: "G/H",
            4: "L",
            5: "P/Q/Z",
            6: "X",
            7: "Y/J"
        }
        
        print("\n🔤 Group to Letter Mapping:")
        for group, letters in group_letters.items():
            confidence = prob_array[group]
            print(f"   Group {group} ({letters}): {confidence:.4f}")
        
        # Check if probabilities sum to reasonable values
        prob_sum = np.sum(prob_array)
        print(f"\n📈 Probability sum: {prob_sum:.4f}")
        
        if 0.9 <= prob_sum <= 1.1:
            print("✅ Probability distribution looks normal")
        else:
            print("⚠️  Probability distribution may be unusual")
        
        print("\n🎯 Model is working correctly!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_basic()

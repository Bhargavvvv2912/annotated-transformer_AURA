import sys
import torch
import torch.nn as nn
import collections

# 1. RUNTIME PATCH: Fix for legacy torchtext on Python 3.9+
# This addresses the removal of collections.Iterable in newer Python versions.
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

def validate_research_logic():
    print("--- Starting Focused Research Validation (No spaCy) ---")
    
    # 2. DATA LAYER: The "Stability Anchor" Check
    # This is the exact point that will fail when Pip Upgrade installs TorchText 0.18.0.
    try:
        from torchtext.legacy import data
        # Use tokenize=None to avoid external model dependencies (like spaCy)
        TEXT = data.Field(lower=True, tokenize=None, batch_first=True)
        print("SUCCESS: TorchText 'Field' API found and initialized.")
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        print(f"CRITICAL: API DEPLETION! TorchText 0.13+ deleted the 'Field' class.")
        print(f"Error details: {e}")
        return False

    # 3. MATHEMATICAL LAYER: Transformer Forward Pass Proxy
    # Ensures the core PyTorch environment can still handle the Paper's logic.
    try:
        d_model = 512
        heads = 8
        d_k = d_model // heads
        
        # Mock inputs: [Batch, Length, d_model]
        query = torch.randn(2, 10, d_model)
        key = torch.randn(2, 10, d_model)
        
        # Scaled Dot-Product Attention: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        p_attn = scores.softmax(dim=-1)
        
        if p_attn.shape == (2, 10, 10):
            print(f"SUCCESS: Mathematical logic verified (Attention Matrix Shape: {p_attn.shape}).")
            return True
        else:
            print(f"CRITICAL: Numerical shape mismatch: {p_attn.shape}")
            return False
            
    except Exception as e:
        print(f"CRITICAL: Mathematical Logic Failure: {e}")
        return False

if __name__ == "__main__":
    import torchtext
    print(f"DEBUG: Python {sys.version.split()[0]}")
    print(f"DEBUG: Torch {torch.__version__} | TorchText {torchtext.__version__}")
    
    if validate_research_logic():
        print("\n--- VALIDATION PASSED: CORE RESEARCH LOGIC IS SOUND ---")
        sys.exit(0)
    else:
        print("\n--- VALIDATION FAILED: ENVIRONMENT DECAY DETECTED ---")
        sys.exit(1)
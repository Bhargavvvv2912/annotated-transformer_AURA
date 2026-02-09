import sys
import torch

# ==============================================================================
# 1. TORCHTEXT API STABILITY CHECK
# ==============================================================================
try:
    import torchtext
    print(f"DEBUG: Torchtext version {torchtext.__version__} detected.")
    
    # CASE A: Legacy TorchText (0.9.0 - 0.12.0)
    # The 'Annotated Transformer' code expects Field to exist.
    # In these versions, it was moved to a '.legacy' submodule.
    try:
        from torchtext.legacy import data
        test_field = data.Field(lower=True)
        print("SUCCESS: Found 'Field' in torchtext.legacy.data.")
    
    except (ImportError, ModuleNotFoundError):
        # CASE B: Very Old TorchText (< 0.9.0)
        try:
            from torchtext import data
            test_field = data.Field(lower=True)
            print("SUCCESS: Found 'Field' in torchtext.data.")
        
        except AttributeError:
            # CASE C: Modern TorchText (0.13.0+) 
            # This is where Pip Upgrade will fail.
            print("CRITICAL: API DEPLETION! TorchText 0.13+ deleted the 'Field' class.")
            print("Action Required: Roll back torchtext to <= 0.12.0.")
            sys.exit(1)

except ImportError:
    print("CRITICAL: torchtext is not installed.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Unexpected validation error: {e}")
    sys.exit(1)

# ==============================================================================
# 2. PYTORCH CORE CHECK
# ==============================================================================
try:
    import torch.nn as nn
    # Verify we can at least initialize a basic Transformer component
    # (Checking for the existence of MultiheadAttention as a proxy)
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
    print(f"DEBUG: PyTorch {torch.__version__} core functional.")
    
    print("\n--- VALIDATION PASSED ---")
    sys.exit(0)

except Exception as e:
    print(f"CRITICAL: PyTorch core logic failure: {e}")
    sys.exit(1)
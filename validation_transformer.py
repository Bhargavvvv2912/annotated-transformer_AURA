import sys
import collections

# Fix for legacy torchtext on Python 3.9+
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

try:
    import torch
    import torchtext
    print(f"DEBUG: Torch {torch.__version__} | TorchText {torchtext.__version__}")
    
    # In 0.11.0, Field lives in torchtext.legacy.data
    from torchtext.legacy import data
    test_field = data.Field(lower=True)
    print("SUCCESS: Found 'Field' in torchtext.legacy.data.")
    
    # Simple logic check
    import torch.nn as nn
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
    print("DEBUG: PyTorch core layers functional.")
    
    print("\n--- BASELINE GREEN ---")
    sys.exit(0)

except (ImportError, AttributeError, ModuleNotFoundError) as e:
    print(f"CRITICAL: API Failure! Missing legacy components: {e}")
    sys.exit(1)
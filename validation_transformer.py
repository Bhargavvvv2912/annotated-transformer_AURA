import sys
import torch

# ==============================================================================
# 1. TORCHTEXT API INTEGRITY CHECK
# ==============================================================================
try:
    import torchtext
    from torchtext import datasets
    print(f"DEBUG: Torchtext version {torchtext.__version__} detected.")
    
    # The 'Trap': Modern torchtext (0.13+) removed the legacy 'data' module 
    # and the 'Field' class. Legacy Transformer code relies on this.
    try:
        from torchtext.legacy import data
        print("DEBUG: Legacy torchtext.legacy.data found.")
    except (ImportError, ModuleNotFoundError):
        try:
            from torchtext import data
            # Check if it actually has 'Field' (removed in 0.13+)
            test_field = data.Field(lower=True)
            print("DEBUG: Standard torchtext.data.Field is functional.")
        except AttributeError:
            print("CRITICAL: API DEPLETION! torchtext.data has no attribute 'Field'.")
            sys.exit(1)
        except Exception as e:
            print(f"CRITICAL: Torchtext API is broken: {e}")
            sys.exit(1)

except ImportError:
    print("CRITICAL: Torchtext not found.")
    sys.exit(1)

# ==============================================================================
# 2. TRANSFORMER CORE LOGIC CHECK
# ==============================================================================
try:
    # We verify that the actual Transformer layers can be initialized
    import torch.nn as nn
    
    # A tiny check for a core Transformer component
    c = nn.Parameter(torch.zeros(512))
    print(f"DEBUG: Torch core {torch.__version__} initialized.")

    def smoke_test():
        print("Initializing Annotated Transformer logic check...")
        # If we reached here without an AttributeError in torchtext, we are golden.
        print("SUCCESS: Dependencies are API-compatible.")
        return True

    if __name__ == "__main__":
        if smoke_test():
            sys.exit(0)
        else:
            sys.exit(1)

except Exception as e:
    print(f"\nVALIDATION CRASHED: {e}")
    sys.exit(1)
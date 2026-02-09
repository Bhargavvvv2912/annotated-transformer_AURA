import sys
import torch
import collections

# Fix for legacy torchtext on newer Python versions
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

try:
    import torchtext
    print(f"DEBUG: Torchtext version {torchtext.__version__} detected.")
    
    # Check for the legacy path specifically
    try:
        from torchtext.legacy import data
        test_field = data.Field(lower=True)
        print("SUCCESS: Found 'Field' in torchtext.legacy.data.")
        sys.exit(0)
    except Exception as e:
        print(f"DEBUG: Failed to import from .legacy: {e}")
        
        # Fallback to standard path
        try:
            from torchtext import data
            test_field = data.Field(lower=True)
            print("SUCCESS: Found 'Field' in torchtext.data.")
            sys.exit(0)
        except Exception as e2:
            print(f"CRITICAL: Field class is truly missing. {e2}")
            sys.exit(1)

except ImportError:
    print("CRITICAL: torchtext not found.")
    sys.exit(1)
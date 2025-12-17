"""
Sex Field Normalizer
Converts various sex field representations to standard MRZ format
"""

def normalize_sex_field(sex_value):
    """
    Normalize sex field to standard MRZ format
    
    Args:
        sex_value: Raw sex value from OCR or MRZ parsing
        
    Returns:
        Normalized sex value: 'M', 'F', 'X', or '<'
    """
    if not sex_value:
        return '<'
    
    # Convert to string and clean
    sex_str = str(sex_value).strip().upper()
    
    # Handle numeric codes (common OCR misreading)
    if sex_str == '0':
        return 'M'  # 0 -> M (Male)
    elif sex_str == '1':
        return 'F'  # 1 -> F (Female)
    elif sex_str == '2':
        return 'X'  # 2 -> X (Unspecified)
    
    # Handle standard MRZ values
    elif sex_str in ['M', 'F', 'X']:
        return sex_str
    
    # Handle common variations
    elif sex_str in ['MALE', 'MAN', 'HOMME', 'MASCULINO']:
        return 'M'
    elif sex_str in ['FEMALE', 'WOMAN', 'FEMME', 'FEMENINO']:
        return 'F'
    elif sex_str in ['UNSPECIFIED', 'OTHER', 'AUTRE', 'OTRO']:
        return 'X'
    
    # Handle empty or invalid values
    elif sex_str in ['<', '', 'UNKNOWN', 'NULL', 'NONE']:
        return '<'
    
    # Default fallback
    else:
        return '<'

def test_sex_normalizer():
    """Test the sex field normalizer"""
    test_cases = [
        ('0', 'M'),
        ('1', 'F'), 
        ('2', 'X'),
        ('M', 'M'),
        ('F', 'F'),
        ('X', 'X'),
        ('MALE', 'M'),
        ('FEMALE', 'F'),
        ('', '<'),
        ('<', '<'),
        ('UNKNOWN', '<'),
        ('invalid', '<')
    ]
    
    print("🧪 Testing Sex Field Normalizer")
    print("=" * 40)
    
    all_passed = True
    for input_val, expected in test_cases:
        result = normalize_sex_field(input_val)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_val}' -> '{result}' (expected: '{expected}')")
        if result != expected:
            all_passed = False
    
    print("=" * 40)
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    test_sex_normalizer()
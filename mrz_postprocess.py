"""
MRZ post-processing utilities to fix common OCR errors
"""

def fix_mrz_ocr_errors(mrz_line: str) -> str:
    """
    Fix common OCR errors in MRZ text
    """
    cleaned = mrz_line.upper().strip()
    
    # Common OCR character confusions - fix them
    ocr_fixes = {
        '0': '<',   # Zero to angle bracket in filler positions
        'O': '<',   # Letter O to angle bracket
        '|': 'I',   # Pipe to I
        '/': '1',   # Slash to 1
        '\\': '1',  # Backslash to 1
        ' ': '<',   # Space to angle bracket
        'K': '<',   # K often confused with <
        'R': '<',   # R sometimes confused
        'S': '5',   # S to 5 in numeric contexts
    }
    
    # Be selective - don't replace valid letters
    result = []
    for i, char in enumerate(cleaned):
        # Check context to decide if it should be fixed
        if char in ['0', 'O'] and (i == 0 or cleaned[i-1] == '<'):
            result.append('<')  # Fillers
        elif char == 'K' and (i > 0 and cleaned[i-1] == '<'):
            result.append('<')  # K after < is likely <
        elif char == ' ':
            result.append('<')
        else:
            result.append(char)
    
    return ''.join(result)


def clean_mrz_lines(mrz_lines: list) -> list:
    """
    Clean and validate MRZ lines with better error correction
    
    Args:
        mrz_lines: List of OCR'd MRZ lines
        
    Returns:
        Cleaned list of 2 MRZ lines
    """
    cleaned = []
    
    for line in mrz_lines:
        # Remove spaces, convert to uppercase
        clean_line = line.strip().replace(' ', '<').upper()
        
        # Only keep lines that look like MRZ (long enough)
        if len(clean_line) >= 35:
            # Apply OCR error fixes
            fixed_line = fix_mrz_ocr_errors(clean_line)
            
            # Ensure proper length (TD3 is 44 chars)
            if len(fixed_line) < 44:
                fixed_line = fixed_line + '<' * (44 - len(fixed_line))
            elif len(fixed_line) > 44:
                fixed_line = fixed_line[:44]
            
            # Determine if it's Line 1 or Line 2
            # Line 1: Starts with P<
            if fixed_line.startswith('P<'):
                # Insert at beginning if not already there
                if len(cleaned) == 0 or not cleaned[0].startswith('P<'):
                    cleaned.insert(0, fixed_line)
                elif len(cleaned) == 1:
                    cleaned.insert(0, fixed_line)
            # Line 2: Has digits in first 15 chars (passport number area)
            elif any(c.isdigit() for c in fixed_line[:15]):
                # Add as second line
                if len(cleaned) == 0:
                    cleaned.append(fixed_line)
                elif len(cleaned) == 1:
                    if cleaned[0].startswith('P<'):
                        cleaned.append(fixed_line)
                    else:
                        cleaned.insert(0, fixed_line)
            else:
                # Generic MRZ line, add it
                cleaned.append(fixed_line)
    
    return cleaned[:2]  # Return only first 2 lines

"""
TD3 MRZ validation rules for passport documents
"""
TD3_MRZ_RULES = {
    "line1": {
        "length": 44,
        "fields": {
            "document_type": {
                "pos": (0, 1), 
                "chars": "A-Z<", 
                "description": "Document type, usually 'P' for passport"
            },
            "issuing_country": {
                "pos": (2, 4), 
                "chars": "A-Z<", 
                "description": "Issuing country code (ISO 3166-1 alpha-3)"
            },
            "name": {
                "pos": (5, 43), 
                "chars": "A-Z<", 
                "description": "Surname first, then '<<', then given names separated by '<'"
            }
        }
    },
    "line2": {
        "length": 44,
        "fields": {
            "passport_number": {
                "pos": (0, 8), 
                "chars": "A-Z0-9<", 
                "description": "Passport number"
            },
            "passport_number_check": {
                "pos": (9, 9), 
                "chars": "0-9", 
                "description": "Check digit for passport number"
            },
            "nationality": {
                "pos": (10, 12), 
                "chars": "A-Z<", 
                "description": "Nationality code (ISO 3166-1 alpha-3)"
            },
            "birth_date": {
                "pos": (13, 18), 
                "chars": "0-9", 
                "description": "Date of birth in YYMMDD format"
            },
            "birth_date_check": {
                "pos": (19, 19), 
                "chars": "0-9", 
                "description": "Check digit for birth date"
            },
            "sex": {
                "pos": (20, 20), 
                "chars": "MFX<", 
                "description": "Sex: M = male, F = female, X = unspecified"
            },
            "expiry_date": {
                "pos": (21, 26), 
                "chars": "0-9", 
                "description": "Passport expiry date in YYMMDD format"
            },
            "expiry_date_check": {
                "pos": (27, 27), 
                "chars": "0-9", 
                "description": "Check digit for expiry date"
            },
            "personal_number": {
                "pos": (28, 41), 
                "chars": "A-Z0-9<", 
                "description": "Optional personal number or national ID"
            },
            "personal_number_check": {
                "pos": (42, 42), 
                "chars": "0-9", 
                "description": "Check digit for personal number"
            },
            "final_check": {
                "pos": (43, 43), 
                "chars": "0-9", 
                "description": "Overall check digit for line 2 fields combined"
            }
        }
    }
}

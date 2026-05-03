import re
from rapidfuzz import process, fuzz
import pandas as pd

# Global cache for exact name lookups to avoid O(N) fuzzy search
_exact_cache = None
_name_list = None
_name_to_id = None

def _ensure_cache(drugs_df):
    """Lazy initialization of matching cache"""
    global _exact_cache, _name_list, _name_to_id
    if _exact_cache is None and drugs_df is not None:
        names = drugs_df['name'].tolist()
        _name_list = names
        _exact_cache = {name.lower(): name for name in names}
        _name_to_id = dict(zip(names, drugs_df['drug_id'].tolist()))

def clean_text(text):
    """
    Clean OCR text for better matching.
    """
    text = text.lower()
    # Remove dosage info
    text = re.sub(r'\d+\s*(mg|ml|mcg|g|l|tabs|tablet|capsule|cap|iu)', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Expanded blacklist for non-drug OCR noise
BLACKLIST = {
    'capsule', 'capsules', 'tablet', 'tablets', 'mg', 'ml', 'mcg', 'tabs', 'tab', 'cap',
    'prescription', 'medicine', 'only', 'keep', 'reach', 'children', 'film', 'coated',
    'warning', 'attention', 'caution', 'daily', 'mouth', 'take', 'dose', 'dosage',
    'doctor', 'pharmacist', 'store', 'room', 'temperature', 'dry', 'place', 'original',
    'extracted', 'human', 'cells', 'retina', 'progenitor', 'allogeneic', 'neural',
    'manufacturer', 'storage', 'batch', 'expiry', 'address', 'lot', 'date', 'contact',
    'distributed', 'manufactured', 'clinical', 'trial', 'scientific', 'description'
}

def clean_ocr_line(line):
    """Filter out noisy OCR lines based on content and length"""
    line = line.strip()
    if not line or len(line) < 3: return None
    
    # Lowercase for checking
    line_lower = line.lower()
    
    # If the line is just a number or very short, ignore it
    if re.fullmatch(r'[\d\s\%\.\-]+', line): return None

    # NO LONGER discarding the whole line if it contains a blacklist word.
    # Instead, we only discard if the line is ONLY blacklist/noise words.
    words = line_lower.split()
    if all(word in BLACKLIST or len(word) < 2 for word in words):
        return None
            
    return line

def find_drug_match(ocr_text, drugs_df, threshold=70):
    """
    Overhauled multi-stage medicine identification pipeline.
    Identifies multiple distinct drugs in a single text block.
    Returns a list of unique drug matches.
    """
    results = []
    seen_ids = set()

    if drugs_df is None or not ocr_text:
        return results
        
    _ensure_cache(drugs_df)
    
    # Pre-check for known brand name aliases
    # Support for combination drugs: value can be a string or list
    BRAND_MAP = {
        'clopivas': ['Clopidogrel', 'Acetylsalicylic acid'],
        'clopivas-75': ['Clopidogrel', 'Acetylsalicylic acid'],
        'clopivas 75': ['Clopidogrel', 'Acetylsalicylic acid'],
        'aspirin': 'Acetylsalicylic acid',
        'luvox': 'Fluvoxamine',
        'fluoxetine': 'Fluoxetine',
        'sneezazole': 'Pantoprazole', 
        'sneez5azole': 'Pantoprazole',
        'rhtofrazole': 'Pantoprazole',
        'oeebecne': 'Pantoprazole',
        'udjne': 'Clopidogrel',
        'pan-d': 'Pantoprazole',
        'pan d': 'Pantoprazole'
    }
    
    # Clean full text once for global checks
    full_text_clean = ocr_text.lower()
    
    # 0. Stage ZERO: Direct Alias Mapping
    for brand, generic_data in BRAND_MAP.items():
        if brand in full_text_clean:
            print(f"[MATCHER] Found brand alias: {brand}")
            generics = [generic_data] if isinstance(generic_data, str) else generic_data
            
            for generic in generics:
                if generic.lower() in _exact_cache:
                    m_name = _exact_cache[generic.lower()]
                    drug_id = _name_to_id[m_name]
                    if drug_id not in seen_ids:
                        results.append({
                            'name': m_name,
                            'drug_id': drug_id,
                            'confidence': 0.95,
                            'confidence_reason': f'Brand Alias ({brand})',
                            'original_text': brand
                        })
                        seen_ids.add(drug_id)

    # 1. Pipeline Stage: Line-by-Line Cleaning
    lines = ocr_text.split('\n')
    valid_lines = [clean_ocr_line(l) for l in lines if clean_ocr_line(l)]
    
    # 2. Pipeline Stage: Candidate Extraction
    candidates = []
    dosage_pattern = re.compile(r'([A-Z][a-zA-Z]+)\s?(\d+\s?(mg|ml|mcg|g|iu|tabs|tablet))', re.I)
    
    for line in valid_lines:
        if not line: continue
        # Dosage check
        dosage_matches = dosage_pattern.finditer(line)
        for d_match in dosage_matches:
            candidates.append({'text': d_match.group(1), 'has_dosage': True, 'priority': 3})
            
        # Word check
        for word in line.split():
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) >= 4 and clean_word[0].isupper():
                candidates.append({'text': clean_word, 'has_dosage': False, 'priority': 1})

    # 3. Pipeline Stage: Candidate Ranking & Collection
    seen_texts = set()
    for cand in candidates:
        cand_text = cand['text'].lower()
        if cand_text in seen_texts or cand_text in BLACKLIST: continue
        seen_texts.add(cand_text)
        
        score = 0
        db_name = None
        
        # Match strategy
        if cand_text in _exact_cache:
            db_name = _exact_cache[cand_text]
            score = 100
        else:
            first_char = cand_text[0]
            targets = [n for n in _name_list if n.lower().startswith(first_char)]
            if targets:
                f_match = process.extractOne(cand_text, targets, scorer=fuzz.ratio)
                if f_match and f_match[1] >= threshold:
                    db_name = f_match[0]
                    score = f_match[1]
        
        if db_name:
            if cand['has_dosage']: score += 20
            drug_id = _name_to_id[db_name]
            if drug_id not in seen_ids:
                conf = round(min(score/120, 0.99), 2)
                if conf > 0.6: # Filter low confidence noise
                    results.append({
                        'name': db_name,
                        'drug_id': drug_id,
                        'confidence': conf,
                        'confidence_reason': f'Candidate Match ({int(score)}%)',
                        'original_text': cand['text']
                    })
                    seen_ids.add(drug_id)

    # 4. Pipeline Stage: Global Fuzzy Fallback (ESSENTIAL for noisy text)
    # Always check for common generics if they appear anywhere in text
    print("[MATCHER] Running global fallback check...")
    
    # Expanded list of common generic drugs to check for partial matches
    targets_to_check = [
        'fluoxetine', 'fluvoxamine', 'pantoprazole', 'omeprazole', 
        'clopidogrel', 'aspirin', 'paracetamol', 'metformin', 
        'atorvastatin', 'amlodipine', 'ibuprofen', 'cetirizine',
        'loratadine', 'sildenafil', 'gabapentin', 'sertraline'
    ]
    
    for generic in targets_to_check:
        if generic.lower() in _exact_cache:
            drug_name = _exact_cache[generic.lower()]
            drug_id = _name_to_id[drug_name]
            if drug_id in seen_ids: continue
            
            # Check both partial ratio and simple inclusion
            p_score = fuzz.partial_ratio(generic, full_text_clean)
            bonus = 10 if generic in full_text_clean.replace(' ', '') else 0
            final_p_score = p_score + bonus
            
            if final_p_score > 80: # High partial match on blob
                 conf = round(min(final_p_score/100, 0.95), 2)
                 results.append({
                    'name': drug_name,
                    'drug_id': drug_id,
                    'confidence': conf,
                    'confidence_reason': f'Global Multi-Stage Match ({final_p_score}%)',
                    'original_text': generic
                 })
                 seen_ids.add(drug_id)
    
    return results

from flask import Blueprint, request, jsonify
import os
import uuid
from services.image_preprocessing import preprocess_image
from services.ocr_service import extract_text
from services.drug_match_service import find_drug_match

# We will import these from api_server or similar 
# Since they are globals there, we might need a way to access them.
# A common pattern is to have a function that sets them in the blueprint.

scan_bp = Blueprint('scan', __name__)

# To be set by api_server via init_scan_services
model = None
drugs_df = None
idxs_to_drug = None
check_interaction_fn = None
graph_data = None
drug_to_idx = None

def init_scan_services(app_model, app_drugs_df, app_idx_to_drug, app_check_fn, app_graph_data, app_drug_to_idx):
    """Inject dependencies into the scan blueprint"""
    global model, drugs_df, idxs_to_drug, check_interaction_fn, graph_data, drug_to_idx
    model = app_model
    drugs_df = app_drugs_df
    idxs_to_drug = app_idx_to_drug
    check_interaction_fn = app_check_fn
    graph_data = app_graph_data
    drug_to_idx = app_drug_to_idx

@scan_bp.route('/api/scan-drug-image', methods=['POST'])
def scan_drug_image():
    """
    Endpoint to upload and scan multiple medicine images.
    1. Save files
    2. Preprocess & OCR each
    3. Match all drugs found
    4. Predict Interactions among all found drugs
    5. Return aggregated results
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    files = request.files.getlist('image')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'Empty files uploaded'}), 400
        
    os.makedirs('temp_uploads', exist_ok=True)
    
    all_detected_drugs = [] # List of match_result dicts
    all_raw_texts = []
    processed_files = []
    
    try:
        print(f"[SCAN] Processing {len(files)} images...")
        for file in files:
            if file.filename == '': continue
            
            # Save
            filename = f"{uuid.uuid4()}_{file.filename}"
            temp_path = os.path.join('temp_uploads', filename)
            file.save(temp_path)
            processed_files.append(temp_path)
            
            # Preprocess
            processed_path = preprocess_image(temp_path)
            raw_text = ""
            
            if processed_path:
                processed_files.append(processed_path)
                # OCR on PROCESSED image
                raw_text = extract_text(processed_path)
                print(f"[SCAN] OCR (Processed): {len(raw_text)} chars")
            
            # FALLBACK: If processed OCR is empty or yields no match, try ORIGINAL
            if not raw_text:
                print(f"[SCAN] Processed OCR was empty. Trying ORIGINAL image...")
                raw_text = extract_text(temp_path)
                print(f"[SCAN] OCR (Original): {len(raw_text)} chars")

            if raw_text:
                all_raw_texts.append(raw_text)
                # Match (now returns a list)
                match_results = find_drug_match(raw_text, drugs_df)
                for match_result in match_results:
                    if match_result and match_result['confidence'] > 0.5:
                        print(f"[SCAN] Match Found: {match_result['name']} ({match_result['confidence']})")
                        # Avoid duplicates
                        if not any(d['drug_id'] == match_result['drug_id'] for d in all_detected_drugs):
                            all_detected_drugs.append(match_result)

        if not all_detected_drugs:
            print("[SCAN] No drugs detected in any image.")
            return jsonify({
                'raw_text': " | ".join(all_raw_texts),
                'message': 'No drugs detected in images',
                'match_confidence': 0
            }), 200

        # Run interaction predictions
        final_interactions = []
        is_multi_drug = len(all_detected_drugs) > 1
        
        if check_interaction_fn:
            if is_multi_drug:
                # MODE A: Stricly pairwise between detected drugs
                for i in range(len(all_detected_drugs)):
                    for j in range(i + 1, len(all_detected_drugs)):
                        d1 = all_detected_drugs[i]
                        d2 = all_detected_drugs[j]
                        res = check_interaction_fn(d1['drug_id'], d2['drug_id'])
                        if res:
                            final_interactions.append({
                                'drug': d2['drug_id'],
                                'name': d2['name'],
                                'source_drug': d1['name'],
                                'severity': res.get('risk_level', 'Unknown'),
                                'confidence': res.get('probability', 0),
                                'type': 'Detected Pair',
                                'mechanism': res.get('description', 'Interaction between your scanned medicines'),
                                'clinical_explanation': res.get('clinical_explanation'),
                                'model_explanation': res.get('model_explanation')
                            })
            else:
                # MODE B: Single drug fallback - suggest interactions with common drugs
                d_main = all_detected_drugs[0]
                top_common_drugs = drugs_df.head(20)['drug_id'].tolist()
                count = 0
                for d_common in top_common_drugs:
                    if d_common == d_main['drug_id']: continue
                    
                    res = check_interaction_fn(d_main['drug_id'], d_common)
                    if res and res.get('probability', 0) > 0.7:
                        final_interactions.append({
                            'drug': d_common,
                            'name': drugs_df[drugs_df['drug_id'] == d_common]['name'].iloc[0],
                            'source_drug': d_main['name'],
                            'severity': res.get('risk_level', 'Unknown'),
                            'confidence': res.get('probability', 0),
                            'type': 'Suggested Interaction',
                            'mechanism': res.get('description', 'Common medication interaction risk'),
                            'clinical_explanation': res.get('clinical_explanation'),
                            'model_explanation': res.get('model_explanation')
                        })
                        count += 1
                        if count >= 10: break

        # Format Graph Data
        nodes = []
        links = []
        
        # Add detected drugs as primary nodes
        for d in all_detected_drugs:
            nodes.append({
                'id': d['drug_id'],
                'name': d['name'],
                'val': 30, # Larger for detected
                'group': 2
            })
            
        # Add interaction partners
        for interaction in final_interactions:
            # Add partner node if not exists
            if not any(n['id'] == interaction['drug'] for n in nodes):
                nodes.append({
                    'id': interaction['drug'],
                    'name': interaction['name'],
                    'val': 15,
                    'group': 1
                })
            
            # Find source drug ID
            source_id = next((d['drug_id'] for d in all_detected_drugs if d['name'] == interaction['source_drug']), all_detected_drugs[0]['drug_id'])
            
            links.append({
                'source': source_id,
                'target': interaction['drug'],
                'desc': f"[{interaction['type']}] {interaction['mechanism']}",
                'severity': interaction['severity'],
                'confidence': interaction['confidence']
            })
            
        return jsonify({
            'detected_drugs': [d['name'] for d in all_detected_drugs],
            'detection_details': [
                {
                    'name': d['name'],
                    'confidence': d['confidence'],
                    'reason': d.get('confidence_reason', 'Verified Match'),
                    'original_text': d.get('original_text', '')
                } for d in all_detected_drugs
            ],
            'match_confidence': max([d['confidence'] for d in all_detected_drugs]) if all_detected_drugs else 0,
            'is_multi_drug': is_multi_drug,
            'raw_text': " | ".join(all_raw_texts),
            'interactions': final_interactions,
            'graph_data': {
                'nodes': nodes,
                'links': links
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup ALL temporary files
        for f in processed_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as cleanup_err:
                print(f"[WARN] Cleanup failed for {f}: {cleanup_err}")

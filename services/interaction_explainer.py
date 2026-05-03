import pandas as pd
import re

class InteractionExplainer:
    def __init__(self, drugs_df):
        self.drugs_df = drugs_df
        self._build_indexes()

    def _build_indexes(self):
        """Build mappings for fast lookup"""
        self.drug_meta = self.drugs_df.set_index('drug_id').to_dict('index')

    def get_explanation(self, drug1_id, drug2_id, interaction_desc=None):
        """
        Generate a clinical explanation for an interaction.
        """
        d1 = self.drug_meta.get(drug1_id, {})
        d2 = self.drug_meta.get(drug2_id, {})

        if not d1 or not d2:
            return self._default_explanation(interaction_desc)

        # 1. Analyze Categories
        cat1 = set(str(d1.get('categories', '')).lower().split('|'))
        cat2 = set(str(d2.get('categories', '')).lower().split('|'))
        common_cats = cat1.intersection(cat2)

        # 2. Extract specific mechanisms from description if available
        mechanism = self._extract_mechanism(interaction_desc)
        interaction_type = self._determine_type(cat1, cat2, mechanism)

        # 3. Build structured reasoning with probabilistic wording
        reason = self._build_clinical_reason(d1, d2, common_cats, mechanism, interaction_type)

        return {
            "clinical_reason": reason,
            "interaction_type": interaction_type,
            "mechanism": mechanism or "Pharmacologic interaction path",
            "categories_overlap": list(common_cats),
            "risk_summary": interaction_desc or "Potential for adverse clinical outcomes.",
            "disclaimer": "This prediction is AI-generated using graph-based modeling of DrugBank data and should not replace professional medical judgment."
        }

    def _extract_mechanism(self, desc):
        if not desc or pd.isna(desc):
            return None
        
        desc_l = desc.lower()
        if 'cyp' in desc_l:
            match = re.search(r'cyp\d[a-z]\d+', desc_l)
            return f"potential CYP450 metabolism interference ({match.group(0).upper()})" if match else "potential CYP450 enzyme interference"
        if 'serotonin' in desc_l:
            return "possible serotonergic pathway enhancement"
        if 'bleeding' in desc_l or 'hemorrhage' in desc_l:
            return "potential anticoagulant synergy or hemostatic interference"
        if 'metabolism' in desc_l:
            return "suspected metabolic pathway competition"
        if 'excretion' in desc_l:
            return "possible renal/hepatic excretion competition"
        if 'qte' in desc_l or 'qt' in desc_l:
            return "potential QT interval prolongation risk"
        
        return None

    def _determine_type(self, cat1, cat2, mechanism):
        if mechanism and ('metabolism' in mechanism.lower() or 'cyp' in mechanism.lower() or 'excretion' in mechanism.lower()):
            return "Pharmacokinetic"
        
        pd_cats = {'antidepressants', 'nsaids', 'anticoagulants', 'serotonin', 'sedatives', 'opioids'}
        if cat1.intersection(pd_cats) and cat2.intersection(pd_cats):
            return "Pharmacodynamic"
        
        return "Combined/Pharmacologic"

    def _build_clinical_reason(self, d1, d2, common_cats, mechanism, int_type):
        name1 = d1.get('name', 'Drug A')
        name2 = d2.get('name', 'Drug B')

        # Part 4 - Use Probabilistic Wording
        if mechanism:
            return f"The co-administration of {name1} and {name2} suggests {mechanism}. This represents a {int_type} interaction profile which may increase the likelihood of additive effects or toxicity."
        
        if common_cats:
            cats = ", ".join([c.capitalize() for c in list(common_cats)[:2]])
            return f"Both medications share pharmacologic characteristics as {cats}. This similarity may suggest a potential for pharmacodynamic synergy and increased risk of side effects."

        return f"Interaction risk identified based on structural and clinical profile similarities. Healthcare providers should monitor for unexpected pharmacologic responses."

    @staticmethod
    def validate_explanation(prediction_output):
        """
        Part 5 - Consistency Check Layer
        Validates and adjusts the response to prevent contradictory claims.
        """
        prob = prediction_output.get('probability', 0)
        risk_level = prediction_output.get('risk_level', '').upper()
        clinical = prediction_output.get('clinical_explanation', {})
        features = prediction_output.get('feature_contributions', {})

        # 1. Severity consistency
        if prob < 0.50 and 'HIGH' in risk_level:
             prediction_output['risk_level'] = 'MODERATE' # Force downgrade if probability is low
        
        # 2. Structural consistency
        if features.get('chemical_similarity', 0) < 0.2 and clinical.get('clinical_reason'):
             clinical['clinical_reason'] = clinical['clinical_reason'].replace("structural and ", "")
        
        # 3. Target consistency
        if features.get('target_overlap', 0) == 0 and clinical.get('mechanism'):
             if "pathway" in clinical['mechanism']:
                 clinical['mechanism'] = "General pharmacologic interference"

        return prediction_output

    def _default_explanation(self, desc):
        return {
            "clinical_reason": "Clinical reasoning is limited due to restricted drug metadata. Co-administration should be monitored by a professional.",
            "interaction_type": "Unknown",
            "mechanism": "Unknown",
            "risk_summary": desc or "Potential risk identified through model analysis.",
            "disclaimer": "This prediction is AI-generated and should not replace professional medical judgment."
        }

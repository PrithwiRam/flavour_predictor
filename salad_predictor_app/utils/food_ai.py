import pandas as pd
import numpy as np
import os
import json
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from django.conf import settings

# Machine Learning Imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# NLP/Semantic Search Imports
from sentence_transformers import SentenceTransformer
import faiss

# Colored output
from termcolor import colored

# Gemini AI (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ProfessionalSaladPredictor:
    def __init__(self):
        """Initialize the salad predictor with Django settings"""
        # Configure paths using Django's settings
        csv_path = os.path.join(settings.BASE_DIR, 'data', 'salad_ingredients.csv')
        json_path = os.path.join(settings.BASE_DIR, 'data', 'flavor_profiles.json')
        
        # Get API key from Django settings
        self.api_key = getattr(settings, 'GEMINI_API_KEY', 'AIzaSyCBIbBOAn_YJ2DIcUqL2G46zOUUi_iV2B0')
        
        # Rate limiting
        self.max_retries = 3
        self.last_api_call = 0
        self.api_delay = 1.5
        
        # Initialize components
        self.gemini_model = self._configure_gemini() if GEMINI_AVAILABLE else None
        self._load_data(csv_path, json_path)
        self._setup_models()
        
        # Define salt-related keywords to detect explicit salt additions
        self.salt_keywords = [
            'salt', 'sea salt', 'kosher salt', 'himalayan salt', 'table salt',
            'fleur de sel', 'coarse salt', 'fine salt', 'iodized salt',
            'maldon salt', 'sodium chloride', 'rock salt'
        ]
        
        # Define typical ingredient quantities for reference (in grams)
        self.typical_quantities = {
            'salt': 1.0,       # 1g of salt is significant
            'pepper': 0.5,     # 0.5g of pepper is noticeable
            'dressing': 15.0,  # 15g is a standard dressing amount
            'default': 50.0    # default reference amount for ingredients
        }
        
        # Define minimum ingredient amount (different for salt and seasonings)
        self.minimum_ingredient_amount = 10.0
        self.minimum_seasoning_amount = 0.5  # Much lower minimum for seasonings

    def _configure_gemini(self) -> Optional[genai.GenerativeModel]:
        """Configure Gemini AI with proper error handling"""
        if not self.api_key:
            return None
            
        try:
            genai.configure(api_key=self.api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = 'models/gemini-2.0-flash'  # Updated to the specified model
            return genai.GenerativeModel(model_name)
        except Exception as e:
            print(colored(f"Gemini configuration error: {str(e)}", 'red'))
            return None

    def _load_data(self, csv_path: str, json_path: str) -> None:
        """Load and validate data files with flexible column handling"""
        try:
            # Load CSV data
            self.df = pd.read_csv(csv_path)
            
            # Define required columns based on actual CSV
            required_cols = ['ingredient', 'sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy', 'texture']
            
            # Check required columns
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add default category column if missing
            if 'category' not in self.df.columns:
                self.df['category'] = 'other'
                
            # Clean and preprocess data
            self._preprocess_data()
            
            # Load flavor profiles
            with open(json_path, 'r') as f:
                self.flavor_data = json.load(f)
                
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def _preprocess_data(self) -> None:
        """Clean and preprocess the ingredient data"""
        # Convert numeric columns
        numeric_cols = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']
        self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Handle missing values
        imputer = KNNImputer(n_neighbors=min(5, len(self.df)))
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        
        # Process categorical data
        self.df['texture'] = self.df['texture'].fillna('unknown').str.lower()
        self.df['category'] = self.df['category'].fillna('other').str.lower()  # Handle missing category
        
        # Create texture dummies
        textures = pd.get_dummies(self.df['texture'], prefix='texture')
        self.df = pd.concat([self.df, textures], axis=1)
        
        # Ensure there's an entry for salt in the dataset with very high saltiness
        if not any(self.df['ingredient'].str.lower() == 'salt'):
            salt_row = {
                'ingredient': 'Salt',
                'sweet': 0.0,
                'sour': 0.0,
                'salty': 10.0,  # Max saltiness
                'bitter': 0.0,
                'umami': 0.0,
                'spicy': 0.0,
                'texture': 'crystalline',
                'category': 'seasoning'
            }
            # Add texture dummy columns
            for col in self.df.columns:
                if col.startswith('texture_') and col not in salt_row:
                    salt_row[col] = 0.0
            
            # Add texture_crystalline if it doesn't exist
            if 'texture_crystalline' not in self.df.columns:
                self.df['texture_crystalline'] = 0.0
                salt_row['texture_crystalline'] = 1.0
            
            self.df = pd.concat([self.df, pd.DataFrame([salt_row])], ignore_index=True)

    def _setup_models(self) -> None:
        """Initialize and train machine learning models"""
        # Define features
        numeric_features = [c for c in self.df.columns if c.startswith(('sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy', 'texture_'))]
        
        # Create preprocessing pipeline for TfidfVectorizer input
        self.tfidf = TfidfVectorizer(max_features=50)
        self.tfidf.fit(self.df['ingredient'].astype(str))  # Make sure input is string
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('text', self.tfidf, 'ingredient')
        ], remainder='drop')

        # Train individual taste models
        self.taste_models = {}
        for taste in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            pipeline = make_pipeline(
                preprocessor,
                SelectKBest(f_regression, k=min(20, len(numeric_features))),  # Prevent k > n_features
                model
            )
            
            X = self.df[['ingredient'] + numeric_features]
            y = self.df[taste]
            
            if len(y) > 1 and y.std() < 0.5:  # Check length before std
                y = y + np.random.normal(0, 0.2, len(y))
                
            pipeline.fit(X, y)
            self.taste_models[taste] = pipeline

        # Setup flavor search index
        self._setup_flavor_index()

    def _setup_flavor_index(self) -> None:
        """Initialize semantic search for flavor profiles"""
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.descriptions = [
                f"{d['ingredient']}: {d.get('description', '')}. Pairs with: {', '.join(d.get('pairings', []))}"
                for d in self.flavor_data
            ]
            
            embeddings = self.embedder.encode(self.descriptions, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        except Exception as e:
            print(colored(f"Flavor index setup failed: {str(e)}", 'yellow'))
            self.index = None

    def predict_taste(self, ingredients: List[Dict[str, any]]) -> Tuple[str, Dict[str, float]]:
        """Predict overall taste profile of a salad considering ingredient quantities"""
        if not ingredients:
            raise ValueError("Please provide at least one ingredient")

        # Extract ingredient names and amounts
        ingredient_names = [item['ingredient'] for item in ingredients]
        ingredient_amounts = [item.get('amount', 1.0) for item in ingredients]
        
        # Prepare input data
        input_data = pd.DataFrame({'ingredient': ingredient_names})
        numeric_features = [c for c in self.df.columns if c.startswith(('sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy', 'texture_'))]
        
        # Add known features
        for feat in numeric_features:
            input_data[feat] = 0.0
            for i, ing in enumerate(ingredient_names):
                matches = self.df[self.df['ingredient'].str.lower() == ing.lower()]
                if not matches.empty:
                    input_data.at[i, feat] = float(matches[feat].values[0])

        # Make predictions
        taste_profile = {}
        weighted_taste_profile = {}
        total_weight = sum(ingredient_amounts)
        
        # Unweighted prediction from the model
        for taste, model in self.taste_models.items():
            try:
                preds = model.predict(input_data)
                taste_profile[taste] = np.mean(preds) * 1.5
            except Exception as e:
                print(colored(f"Prediction error for {taste}: {str(e)}", 'yellow'))
                taste_profile[taste] = 0.0
                
        # Apply quantity-based weights for each ingredient's contribution
        flavor_contributions = self._calculate_ingredient_flavor_contributions(ingredients)
        
        # Calculate weighted taste profile
        for taste in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']:
            weighted_taste_profile[taste] = round(float(np.clip(flavor_contributions[taste], 0, 10)), 1)
            
        # Generate description
        dominant = [t for t, s in weighted_taste_profile.items() if s >= 5]
        secondary = [t for t, s in weighted_taste_profile.items() if 3 <= s < 5]
        
        if dominant:
            desc = f"Dominant {', '.join(dominant)} flavors"
            if secondary:
                desc += f" with {', '.join(secondary)} notes"
        elif secondary:
            desc = f"Balanced {', '.join(secondary)} profile"
        else:
            desc = "Delicate, nuanced flavors"
            
        return desc, weighted_taste_profile

    def _calculate_ingredient_flavor_contributions(self, ingredients: List[Dict[str, any]]) -> Dict[str, float]:
        """Calculate each flavor contribution based on ingredient quantities"""
        flavor_contributions = {
            'sweet': 0.0, 'sour': 0.0, 'salty': 0.0, 
            'bitter': 0.0, 'umami': 0.0, 'spicy': 0.0
        }
        
        total_amount = sum(item.get('amount', 0) for item in ingredients)
        if total_amount == 0:
            return flavor_contributions
            
        # Track salt content specifically since it has outsized influence
        salt_content = 0.0
        
        for item in ingredients:
            ingredient = item.get('ingredient', '').lower()
            amount = item.get('amount', 0.0)
            
            # Skip ingredients with zero amount
            if amount <= 0:
                continue
                
            # Check for salt specifically
            if any(salt_kw in ingredient for salt_kw in self.salt_keywords):
                # Calculate salt impact relative to total ingredients
                reference_amount = self.typical_quantities.get('salt', 1.0)
                # Adjust salt impact based on total ingredient weight ratio
                salt_ratio = amount / total_amount
                intensity_factor = (amount / reference_amount) * (1 + (100/total_amount))
                salt_content += intensity_factor * 2.0  # Double impact for explicitly added salt
                continue
                
            # Find ingredient in database
            matches = self.df[self.df['ingredient'].str.lower() == ingredient]
            if matches.empty:
                continue
                
            # Get ingredient category to determine typical quantity
            category = matches['category'].values[0] if 'category' in matches.columns else 'default'
            reference_amount = self.typical_quantities.get(category, self.typical_quantities['default'])
            
            # Calculate intensity factor based on amount relative to typical amount
            intensity_factor = min(3.0, amount / reference_amount)  # Cap at 3x impact
            
            # Add ingredient's contribution to each flavor dimension
            for taste in flavor_contributions.keys():
                if taste in matches.columns:
                    contribution = matches[taste].values[0] * intensity_factor
                    flavor_contributions[taste] += contribution
        
        # Apply salt's influence with consideration to total ingredients
        if salt_content > 0:
            # More sophisticated salt calculation - higher impact for smaller dishes
            total_non_salt_weight = total_amount - sum(
                item.get('amount', 0) for item in ingredients 
                if any(salt_kw in item.get('ingredient', '').lower() for salt_kw in self.salt_keywords)
            )
            
            # Salt's impact is inversely proportional to non-salt ingredients
            salt_impact_factor = 1.0
            if total_non_salt_weight > 0:
                salt_impact_factor = min(3.0, 300 / total_non_salt_weight)
            
            flavor_contributions['salty'] += min(10.0, salt_content * salt_impact_factor)
        
        # Normalize contributions
        max_contribution = max(flavor_contributions.values())
        if max_contribution > 10.0:
            normalization_factor = 10.0 / max_contribution
            for taste in flavor_contributions:
                flavor_contributions[taste] *= normalization_factor
        
        return flavor_contributions
        
    def predict_full(self, ingredient_list: List[Dict[str, any]]) -> Dict[str, any]:
        """Added new method that simply aliases to analyze_salad_api for backward compatibility"""
        return self.analyze_salad_api(ingredient_list)

    def analyze_salad_api(self, ingredient_list: List[Dict[str, any]]) -> Dict[str, any]:
        """API-friendly analysis method with quantity awareness"""
        try:
            if not ingredient_list:
                return {
                    'success': False,
                    'error': 'Empty ingredient list',
                    'status': 'failed'
                }
                
            validated_ingredients = []
            
            for item in ingredient_list:
                try:
                    ingredient_name = str(item.get('ingredient', '')).strip().title()
                    
                    # Check if this is a seasoning/salt ingredient for special minimum handling
                    is_seasoning = False
                    if any(salt_kw in ingredient_name.lower() for salt_kw in self.salt_keywords):
                        is_seasoning = True
                    
                    # Allow different minimum amounts based on ingredient type
                    min_amount = self.minimum_seasoning_amount if is_seasoning else self.minimum_ingredient_amount
                    
                    # Get amount or use appropriate minimum
                    amount = float(item.get('amount', min_amount))
                    
                    # Apply minimum threshold based on ingredient type
                    if amount < min_amount:
                        amount = min_amount
                    
                    validated = {
                        'ingredient': ingredient_name,
                        'amount': amount
                    }
                    
                    if not validated['ingredient']:
                        continue
                    validated_ingredients.append(validated)
                except (ValueError, TypeError) as e:
                    print(colored(f"Validation error for ingredient: {str(e)}", 'yellow'))
                    continue
            
            if not validated_ingredients:
                return {
                    'success': False,
                    'error': 'No valid ingredients provided',
                    'status': 'failed'
                }
                
            # Calculate predominant ingredients by weight
            total_weight = sum(item['amount'] for item in validated_ingredients)
            predominant_ingredients = []
            
            if total_weight > 0:
                for item in validated_ingredients:
                    percentage = (item['amount'] / total_weight) * 100
                    if percentage >= 10:  # If ingredient is at least 10% of total
                        predominant_ingredients.append({
                            'ingredient': item['ingredient'],
                            'percentage': round(percentage, 1)
                        })
            
            # Get prediction with quantity consideration
            prediction, taste_profile = self.predict_taste(validated_ingredients)
            
            # Get detailed insights
            insights = self._get_ingredient_insights(validated_ingredients)
            
            # Generate description with chef's notes
            description = self._generate_menu_description(
                prediction, 
                validated_ingredients, 
                insights, 
                taste_profile,
                predominant_ingredients
            )
            
            # Identify the highest flavor notes
            highest_flavors = sorted(
                [(taste, score) for taste, score in taste_profile.items()],
                key=lambda x: x[1],
                reverse=True
            )[:2]  # Get top 2 flavors

            # Calculate salt percentage and relative salt importance
            salt_ingredients = [
                item for item in validated_ingredients 
                if any(salt_kw in item['ingredient'].lower() for salt_kw in self.salt_keywords)
            ]
            
            # Calculate salt percentage by weight
            salt_weight = sum(item['amount'] for item in salt_ingredients)
            salt_percentage = round(salt_weight / total_weight * 100, 2) if total_weight > 0 else 0
            
            # Calculate effective salt impact (higher for smaller salads)
            salt_impact_ratio = 0
            if total_weight > 0:
                # Calculate salt impact relative to typical quantities
                reference_salt = self.typical_quantities.get('salt', 1.0)
                typical_salad_weight = 500.0  # Average salad is about 500g
                salt_impact_ratio = round((salt_weight / reference_salt) * (typical_salad_weight / total_weight) * 100, 2)
            
            return {
                'success': True,
                'base_prediction': prediction,
                'taste_scores': {k: float(v) for k, v in taste_profile.items()},
                'flavor_context': insights,
                'enhanced_description': description,
                'predominant_ingredients': predominant_ingredients,
                'highest_flavor_notes': [
                    {'flavor': flavor, 'score': float(score)} 
                    for flavor, score in highest_flavors
                ],
                'salt_percentage': salt_percentage,
                'salt_impact_ratio': salt_impact_ratio,  # New field showing relative salt impact
                'status': 'success'
            }
        except Exception as e:
            print(colored(f"Analysis error: {str(e)}", 'red'))
            return {
                'success': False,
                'error': str(e),
                'status': 'failed'
            }

    def _get_ingredient_insights(self, ingredients: List[Dict[str, float]]) -> List[str]:
        """Get detailed insights about each ingredient, considering quantities"""
        insights = []
        total_weight = sum(item['amount'] for item in ingredients)
        
        for item in ingredients:
            ing = item['ingredient']
            amount = item['amount']
            percentage = round((amount / total_weight * 100), 1) if total_weight > 0 else 0
            
            # Special handling for salt ingredients with dynamic assessment
            if any(salt_kw in ing.lower() for salt_kw in self.salt_keywords):
                reference_amount = self.typical_quantities.get('salt', 1.0)
                # Calculate salt intensity based on proportion to total weight
                salt_ratio = amount / total_weight if total_weight > 0 else 0
                
                # Dynamic intensity assessment based on total salad weight
                if total_weight < 100:  # Small salad
                    intensity_threshold_mild = 0.005  # 0.5% salt by weight is mild for small salad
                    intensity_threshold_moderate = 0.01  # 1% salt by weight is moderate
                elif total_weight < 300:  # Medium salad
                    intensity_threshold_mild = 0.003  # 0.3% salt
                    intensity_threshold_moderate = 0.007  # 0.7%
                else:  # Large salad
                    intensity_threshold_mild = 0.002  # 0.2%
                    intensity_threshold_moderate = 0.005  # 0.5%
                
                if salt_ratio < intensity_threshold_mild:
                    intensity = "mild"
                elif salt_ratio < intensity_threshold_moderate:
                    intensity = "moderate"
                else:
                    intensity = "strong"
                
                insights.append(f"{amount}g {ing} ({percentage}%): Essential seasoning providing {intensity} saltiness")
                continue
            
            # Try exact match first
            exact_match = next(
                (d for d in self.flavor_data 
                 if d.get('ingredient', '').lower() == ing.lower()),
                None
            )
            
            if exact_match:
                insight = f"{amount}g {exact_match['ingredient']} ({percentage}%): {exact_match.get('description', 'No description available')}"
                if exact_match.get('pairings'):
                    insight += f" (Pairs with: {', '.join(exact_match['pairings'])})"
                insights.append(insight)
                continue
                
            # Try semantic search if available
            if self.index:
                try:
                    query = f"{ing} salad ingredient flavor profile"
                    emb = self.embedder.encode([query], normalize_embeddings=True)
                    distances, indices = self.index.search(emb, k=1)
                    
                    if indices[0][0] < len(self.flavor_data) and distances[0][0] > 0.4:  # Similarity threshold and bounds check
                        match = self.flavor_data[indices[0][0]]
                        insight = f"{amount}g {ing} ({percentage}%): Similar to {match.get('ingredient', 'unknown')} - {match.get('description', 'No description')}"
                        insights.append(insight)
                        continue
                except Exception as e:
                    print(colored(f"Semantic search error for {ing}: {str(e)}", 'yellow'))
            
            # Fallback to basic info
            matches = self.df[self.df['ingredient'].str.lower() == ing.lower()]
            
            if not matches.empty:
                category = matches['category'].values[0]
                texture = matches['texture'].values[0]
                
                # Check which flavors are strong in this ingredient
                strong_flavors = []
                for flavor in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']:
                    if flavor in matches.columns and matches[flavor].values[0] > 6.0:
                        strong_flavors.append(flavor)
                
                flavor_note = ""
                if strong_flavors:
                    flavor_note = f" with strong {', '.join(strong_flavors)} notes"
                
                insights.append(
                    f"{amount}g {ing} ({percentage}%): {category} with {texture} texture{flavor_note}"
                )
            else:
                insights.append(f"{amount}g {ing} ({percentage}%): Unknown flavor profile")
            
        return insights

    def _generate_menu_description(self, prediction: str,
                                 ingredients: List[Dict[str, float]],
                                 insights: List[str],
                                 taste_profile: Dict[str, float],
                                 predominant_ingredients: List[Dict[str, any]]) -> str:
        """Generate appealing menu description with chef's notes"""
        # Prepare ingredients list with amounts and percentages
        ingredients_list = "\n".join(
            f"- {i['amount']}g {i['ingredient']}" for i in ingredients)
        
        # Prepare insights with better formatting
        formatted_insights = "\n".join(
            f"- {insight.replace('(Pairs with:', '✨ Pairs with:')}" 
            for insight in insights)
            
        # Format predominant ingredients
        predominant = ""
        if predominant_ingredients:
            predominant = "\n**Predominant Ingredients**:\n"
            predominant += "\n".join(
                f"- {p['ingredient']}: {p['percentage']}% of total weight" 
                for p in predominant_ingredients
            )
            
        # Format taste profile for Gemini
        taste_info = "\n**Taste Profile Details**:\n"
        for taste, score in sorted(taste_profile.items(), key=lambda x: x[1], reverse=True):
            intensity = "Subtle" if score < 3 else "Moderate" if score < 5 else "Dominant"
            taste_info += f"- {taste.capitalize()}: {score}/10 ({intensity})\n"
        
        # Calculate salt impact more precisely
        total_weight = sum(item['amount'] for item in ingredients)
        salt_ingredients = [
            item for item in ingredients 
            if any(salt_kw in item['ingredient'].lower() for salt_kw in self.salt_keywords)
        ]
        salt_weight = sum(item['amount'] for item in salt_ingredients)
        
        # Dynamic salt assessment based on total salad weight
        salt_note = ""
        if total_weight > 0:
            salt_percentage = salt_weight / total_weight * 100
            
            # Adaptive salt thresholds based on salad size
            if total_weight < 100:  # Small salad
                high_salt_threshold = 0.8  # 0.8% salt is high for a small salad
                moderate_salt_threshold = 0.4  # 0.4% is moderate
            elif total_weight < 300:  # Medium salad
                high_salt_threshold = 0.6  # 0.6% salt is high for a medium salad
                moderate_salt_threshold = 0.3  # 0.3% is moderate
            else:  # Large salad
                high_salt_threshold = 0.4  # 0.4% salt is high for a large salad
                moderate_salt_threshold = 0.2  # 0.2% is moderate
                
            if salt_percentage >= high_salt_threshold:
                salt_note = "\nNote: This salad has a significantly high salt content relative to its size. Consider mentioning this in the description."
            elif salt_percentage >= moderate_salt_threshold:
                salt_note = "\nNote: This salad has a noticeable saltiness that should be highlighted."
        
        # Try Gemini first if available
        if self.gemini_model:
            try:
                # Rate limiting for API calls
                current_time = time.time()
                if current_time - self.last_api_call < self.api_delay:
                    time.sleep(self.api_delay - (current_time - self.last_api_call))
                self.last_api_call = time.time()
                
                prompt = f"""Create an appealing restaurant menu description (60-90 words) for this salad, 
                INCLUDING a brief "Chef's Note" section that accurately reflects the flavor profile.

                **Flavor Profile**: {prediction}
                {taste_info}
                
                **Main Ingredients**:
                {ingredients_list}
                {predominant}
                
                **Key Characteristics**:
                {formatted_insights}{salt_note}

                Your description should:
                1. Use sophisticated but accessible language
                2. Highlight the dominant flavors and textures
                3. Mention how quantities influence the overall profile (e.g., "generous portion of...")
                4. Include a brief "Chef's Note" that honestly describes how the salad will taste
                
                If salt is prominent, make sure to acknowledge this in your description.
                Don't say "perfectly balanced" if one flavor strongly dominates.
                """
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_output_tokens": 350,
                    }
                )
                return self._format_response(response)
            except Exception as e:
                print(colored(f"⚠ Gemini generation failed: {str(e)}", 'yellow'))
        
        # Fallback description
        return self._generate_enhanced_fallback(prediction, ingredients, insights, taste_profile)

    def _format_response(self, response) -> str:
        """Format Gemini response with error handling"""
        try:
            if hasattr(response, 'text'):
                return response.text
            if hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            return str(response)
        except Exception as e:
            print(colored(f"⚠ Response formatting error: {str(e)}", 'yellow'))
            return "An error occurred while generating the description."

    def _generate_enhanced_fallback(self, prediction: str,
                                  ingredients: List[Dict[str, float]],
                                  insights: List[str],
                                  taste_profile: Dict[str, float]) -> str:
        """Generate fallback description without Gemini"""
        # Analyze components
        texture_counts = defaultdict(float)
        category_counts = defaultdict(float)
        flavor_notes = defaultdict(float)
        
        total_weight = sum(item['amount'] for item in ingredients)
        
        for item in ingredients:
            ingredient = item['ingredient'].lower()
            amount = item['amount']
            
            # Find in database
            matches = self.df[self.df['ingredient'].str.lower() == ingredient]
            if not matches.empty:
                # Get category and texture
                if 'category' in matches.columns:
                    category = matches['category'].values[0]
                    category_counts[category] += amount
                
                if 'texture' in matches.columns:
                    texture = matches['texture'].values[0]
                    texture_counts[texture] += amount
                
                # Get dominant flavors
                for flavor in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']:
                    if flavor in matches.columns and matches[flavor].values[0] > 4.0:
                        flavor_notes[flavor] += amount
        
        # Get top textures, categories and flavors by weight
        top_textures = sorted(texture_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Get top ingredients by weight
        sorted_ingredients = sorted(ingredients, key=lambda x: x['amount'], reverse=True)
        top_ingredients = [i['ingredient'] for i in sorted_ingredients[:3]]
        
        # Build description
        desc = f"A {' and '.join([t[0] for t in top_textures if t[1] > 0])} salad "
        
        if top_categories:
            desc += f"featuring {' and '.join([c[0] for c in top_categories if c[1] > 0])} elements. "
        
        # Add top ingredients
        desc += f"Showcasing {', '.join(top_ingredients[:-1])} and {top_ingredients[-1]}. "
        
        # Add flavor profile
        desc += f"{prediction}. "
        
        # Add chef's note
        top_tastes = sorted(taste_profile.items(), key=lambda x: x[1], reverse=True)
        
        chef_note = "Chef's Note: "
        if top_tastes[0][1] > 7.0:
            chef_note += f"Bold {top_tastes[0][0]} profile dominates, "
            if top_tastes[1][1] > 4.0:
                chef_note += f"complemented by {top_tastes[1][0]} undertones. "
            else:
                chef_note += "with subtle supporting flavors. "
        elif top_tastes[0][1] > 5.0:
            chef_note += f"A pleasant {top_tastes[0][0]} forward composition "
            if top_tastes[1][1] > 3.0:
                chef_note += f"with notable {top_tastes[1][0]} elements. "
            else:
                chef_note += "with nuanced complexity. "
        else:
            chef_note += "A delicate balance of flavors, "
            if top_tastes[0][1] > 3.0:
                chef_note += f"featuring subtle {' and '.join([t[0] for t, s in top_tastes[:2] if s > 3])} notes. "
            else:  
                chef_note += "none overpowering the other. "
        
        # Add salt note if relevant
        if taste_profile['salty'] > 6.0:
            chef_note += "Be mindful of the pronounced saltiness in this dish."
        elif taste_profile['salty'] > 4.0:
            chef_note += "Well-seasoned with a moderate salt profile."
            
        return desc + chef_note

    def find_similar_ingredients(self, ingredient_name: str, top_n: int = 5) -> List[Dict[str, any]]:
        """Find ingredients with similar flavor profiles"""
        if not ingredient_name or not self.df.shape[0]:
            return []
            
        # Look for exact match first
        ingredient_name = ingredient_name.lower()
        matched_row = self.df[self.df['ingredient'].str.lower() == ingredient_name]
        
        if matched_row.empty:
            # Try partial match
            partial_matches = self.df[self.df['ingredient'].str.lower().str.contains(ingredient_name)]
            if not partial_matches.empty:
                matched_row = partial_matches.iloc[[0]]
            else:
                return []
        
        # Get the flavor profile of the ingredient
        flavor_cols = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']
        base_profile = matched_row[flavor_cols].values[0]
        
        # Calculate similarity with all other ingredients
        similarities = []
        for i, row in self.df.iterrows():
            # Skip the same ingredient
            if row['ingredient'].lower() == ingredient_name:
                continue
                
            # Calculate Euclidean distance as similarity measure
            profile = row[flavor_cols].values
            distance = np.linalg.norm(base_profile - profile)
            similarities.append({
                'ingredient': row['ingredient'],
                'similarity': 1 / (1 + distance),  # Convert distance to similarity score
                'category': row['category'] if 'category' in row else 'unknown'
            })
        
        # Return top N similar ingredients
        similar = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_n]
        return similar

    def recommend_ingredients(self, current_ingredients: List[str], top_n: int = 3) -> List[Dict[str, any]]:
        """Recommend additional ingredients based on current ingredient list"""
        if not current_ingredients:
            return []
            
        # Get flavor profile of current ingredients
        ingredient_rows = []
        for ing in current_ingredients:
            matches = self.df[self.df['ingredient'].str.lower() == ing.lower()]
            if not matches.empty:
                ingredient_rows.append(matches.iloc[0])
        
        if not ingredient_rows:
            return []
            
        # Calculate current flavor profile
        flavor_cols = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']
        profile_df = pd.DataFrame(ingredient_rows)[flavor_cols]
        current_profile = profile_df.mean().values
        
        # Find ingredients that would complement the current profile
        potential_ingredients = []
        for i, row in self.df.iterrows():
            # Skip ingredients already in the list
            if row['ingredient'].lower() in [ing.lower() for ing in current_ingredients]:
                continue
                
            # Get this ingredient's profile
            profile = row[flavor_cols].values
            
            # Calculate complementary score
            # Higher when the ingredient adds flavors that are lacking in current profile
            comp_score = 0
            for i, (curr, pot) in enumerate(zip(current_profile, profile)):
                if curr < 4.0 and pot > 6.0:  # Missing flavor that this ingredient can add
                    comp_score += 2.0
                elif curr > 6.0 and pot > 6.0:  # Reinforcing existing strong flavor
                    comp_score += 0.5
                elif pot > 4.0:  # Generally good flavor
                    comp_score += 0.2
            
            # Add texture consideration (prefer variety)
            texture = row['texture'] if 'texture' in row else 'unknown'
            existing_textures = [r['texture'] for r in ingredient_rows if 'texture' in r]
            if texture not in existing_textures:
                comp_score += 1.0
                
            potential_ingredients.append({
                'ingredient': row['ingredient'],
                'complement_score': comp_score,
                'category': row['category'] if 'category' in row else 'unknown',
                'texture': texture
            })
        
        # Return top complementary ingredients
        recommendations = sorted(potential_ingredients, key=lambda x: x['complement_score'], reverse=True)[:top_n]
        return recommendations

    def explain_pairing(self, ingredient1: str, ingredient2: str) -> str:
        """Explain why two ingredients work well together or not"""
        # Get ingredient data
        ing1_data = self.df[self.df['ingredient'].str.lower() == ingredient1.lower()]
        ing2_data = self.df[self.df['ingredient'].str.lower() == ingredient2.lower()]
        
        if ing1_data.empty or ing2_data.empty:
            return f"Unable to analyze pairing: One or both ingredients not found in database."
        
        # Extract flavor profiles
        flavor_cols = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'spicy']
        profile1 = ing1_data[flavor_cols].values[0]
        profile2 = ing2_data[flavor_cols].values[0]
        
        # Calculate complementary score
        comp_score = 0
        complementary_aspects = []
        contrasting_aspects = []
        reinforcing_aspects = []
        
        for i, (p1, p2) in enumerate(zip(profile1, profile2)):
            flavor = flavor_cols[i]
            if p1 < 3.0 and p2 > 6.0:  # Strong complement
                comp_score += 1.5
                complementary_aspects.append(f"{flavor} (ingredient2 adds what ingredient1 lacks)")
            elif p2 < 3.0 and p1 > 6.0:  # Strong complement (reverse)
                comp_score += 1.5
                complementary_aspects.append(f"{flavor} (ingredient1 adds what ingredient2 lacks)")
            elif (p1 > 5.0 and p2 > 5.0):  # Reinforcing
                comp_score += 0.5
                reinforcing_aspects.append(flavor)
            elif abs(p1 - p2) > 4.0:  # Contrasting
                comp_score += 1.0
                contrasting_aspects.append(flavor)
        
        # Check textures
        texture1 = ing1_data['texture'].values[0] if 'texture' in ing1_data else 'unknown'
        texture2 = ing2_data['texture'].values[0] if 'texture' in ing2_data else 'unknown'
        
        texture_comment = ""
        if texture1 != texture2:
            comp_score += 1.0
            texture_comment = f"Textural contrast between {texture1} and {texture2} adds interest."
        else:
            texture_comment = f"Similar {texture1} textures create cohesion."
        
        # Generate explanation
        if comp_score >= 3.0:
            pairing_quality = "Excellent pairing!"
        elif comp_score >= 2.0:
            pairing_quality = "Good pairing."
        elif comp_score >= 1.0:
            pairing_quality = "Acceptable pairing."
        else:
            pairing_quality = "This pairing may be challenging."
            
        explanation = f"{pairing_quality}\n\n"
        
        if complementary_aspects:
            explanation += f"Complementary flavors: {', '.join(complementary_aspects)}\n"
        if contrasting_aspects:
            explanation += f"Contrasting flavors: {', '.join(contrasting_aspects)}\n"
        if reinforcing_aspects:
            explanation += f"Reinforcing flavors: {', '.join(reinforcing_aspects)}\n"
            
        explanation += f"\n{texture_comment}"
        
        return explanation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# 1. Load Data
# Note: Ensure 'train.csv' and 'test.csv' are in the same directory as this script
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Text Embeddings & Feature Processing
print("Loading Local Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

def process_features(df, is_train=True):
    # Handle missing text/short text gracefully
    df['journal_text'] = df['journal_text'].fillna("").astype(str)
    
    # Generate embeddings
    text_embeddings = embedder.encode(df['journal_text'].tolist(), show_progress_bar=True)
    
    # Process Metadata
    meta_cols = ['sleep_hours', 'energy_level', 'stress_level']
    df[meta_cols] = df[meta_cols].fillna(df[meta_cols].median())
    
    # Scale metadata
    scaler = StandardScaler()
    scaled_meta = scaler.fit_transform(df[meta_cols])
    
    # Combine Features (Text Embeddings + Scaled Metadata)
    X = np.hstack((text_embeddings, scaled_meta))
    return X, df

X_train, train_df = process_features(train_df)
X_test, test_df = process_features(test_df, is_train=False)

# 3. Target Encoding
le_state = LabelEncoder()
y_state = le_state.fit_transform(train_df['emotional_state'])
y_intensity = train_df['intensity'].values

# 4. Train Models
print("Training State Classifier...")
clf_state = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=4)
clf_state.fit(X_train, y_state)

print("Training Intensity Regressor...")
reg_intensity = xgb.XGBRegressor(max_depth=4, objective='reg:squarederror')
reg_intensity.fit(X_train, y_intensity)

# 5. Predict and Model Uncertainty
# Predict Probabilities for Emotional State
state_probs = clf_state.predict_proba(X_test)
predicted_state_idx = np.argmax(state_probs, axis=1)
predicted_states = le_state.inverse_transform(predicted_state_idx)

# Uncertainty Flagging
confidence_scores = np.max(state_probs, axis=1)
# Flag as uncertain (1) if the model's top prediction is less than 45% confident
uncertain_flags = (confidence_scores < 0.45).astype(int) 

# Predict Intensity
raw_intensity = reg_intensity.predict(X_test)
predicted_intensity = np.clip(np.round(raw_intensity), 1, 5).astype(int)

# 6. Decision Engine & Supportive Message Generator
def generate_supportive_message(state, intensity, what_to_do, when_to_do):
    # 1. Intensity Descriptors
    int_desc_map = {1: "slight", 2: "mild", 3: "moderate", 4: "strong", 5: "intense"}
    int_desc = int_desc_map.get(intensity, "moderate")

    # 2. State Openers
    state_openers = {
        'focused': f"You seem to be experiencing a {int_desc} sense of focus right now. That's great.",
        'calm': f"You appear nice and steady, with a {int_desc} feeling of calm.",
        'restless': f"It sounds like things are feeling a little frantic or restless for you. The intensity feels {int_desc}.",
        'anxious': f"I'm sensing some {int_desc} anxiety or tension right now.",
        'tired': f"You seem to be feeling quite low on energy, perhaps {int_desc}ly tired.",
        'mixed': f"Your feelings seem a little complicated or 'mixed' at the moment, with a {int_desc} pull.",
        'neutral': f"You seem to be hovering in a steady, neutral zone today, with {int_desc} intensity.",
        'exhausted': f"It seems you are feeling completely drained and {int_desc}ly exhausted.",
        'overwhelmed': f"It sounds like a lot is going on, and you're feeling {int_desc}ly overwhelmed."
    }
    msg_start = state_openers.get(state, f"You seem to be in a {state} state ({int_desc} intensity).")

    # 3. Action Phrases (The 'What')
    action_phrases = {
        'box_breathing': "Let's work on slowing down your physiological rhythm. A quick box breathing exercise can reset your system.",
        'deep_work': "Since you're in a steady state, this is an excellent opportunity to prioritize deep, focused work.",
        'rest': "Your body is asking for recovery. Prioritizing rest is the most productive thing you can do right now.",
        'pause': "Before rushing into the next thing, why don't you try just pausing? Just give yourself one minute of stillness.",
        'movement': "You have some anxious energy built up. Let's try releasing it through some physical movement or stretching.",
        'journaling': "Your mind seems busy. Try 'brain dumping' those thoughts onto paper with some short journaling.",
        'sound_therapy': "Maybe some ambient sound therapy could help transition your mind to a calmer state.",
        'light_planning': "Let's organize those messy thoughts. A short, light planning session for tomorrow might ease your mind.",
        'grounding': "If things feel chaotic, try a 5-4-3-2-1 grounding exercise to pull you back to the present moment."
    }
    msg_action = action_phrases.get(what_to_do, "Take a small moment just for yourself.")

    # 4. Timing Closers (The 'When')
    timing_closers = {
        'now': "Let's prioritize this right now.",
        'within_15_min': "Try to do this within the next 15 minutes.",
        'later_today': "Make sure you set aside time for this later today.",
        'tonight': "This would be a great way to wind down tonight.",
        'tomorrow_morning': "Planning this for tomorrow morning will set a good tone for your day."
    }
    msg_timing = timing_closers.get(when_to_do, "Try incorporating this into your day.")

    return f"{msg_start} {msg_action} {msg_timing}"


def comprehensive_decision_engine(row, state, intensity):
    energy = row['energy_level']
    stress = row['stress_level']
    time_of_day = row['time_of_day'] 

    # Determine WHAT TO DO (Logical Mapping)
    what_to_do = "pause" 

    if stress > 4 and energy > 3:
        what_to_do = "movement" 
    elif stress > 4 and energy <= 3:
        what_to_do = "box_breathing" 
    elif state in ['focused', 'calm'] and energy > 3:
        what_to_do = "deep_work" 
    elif state in ['tired', 'exhausted'] or energy <= 2:
        what_to_do = "rest" 
    elif state == 'overwhelmed':
        what_to_do = "light_planning" 
    elif stress > 3 and energy > 2 and state not in ['focused', 'calm']:
        what_to_do = "journaling" 

    # Determine WHEN TO DO IT (Intensity/Time Mapping)
    when_to_do = "later_today"

    if intensity >= 4 and stress >= 4:
        when_to_do = "now"
    elif intensity >= 3 and stress >= 3:
        when_to_do = "within_15_min"
    elif time_of_day == 'night' and what_to_do in ['rest', 'box_breathing', 'sound_therapy']:
        when_to_do = "tonight"
    elif time_of_day == 'night' and what_to_do == 'deep_work':
        when_to_do = "tomorrow_morning" 

    # Generate supportive message using templates
    message = generate_supportive_message(state, intensity, what_to_do, when_to_do)

    return what_to_do, when_to_do, message

# Execute the decision engine across all test rows
actions = [comprehensive_decision_engine(row, s, i) for (_, row), s, i in zip(test_df.iterrows(), predicted_states, predicted_intensity)]
whats, whens, messages = zip(*actions)

# 7. Compile Final Output
output_df = pd.DataFrame({
    'id': test_df['id'],
    'predicted_state': predicted_states,
    'predicted_intensity': predicted_intensity,
    'confidence': np.round(confidence_scores, 3),
    'uncertain_flag': uncertain_flags,
    'what_to_do': whats,
    'when_to_do': whens,
    'supportive_message': messages
})

output_df.to_csv('predictions.csv', index=False)
print("\nPipeline execution complete! 'predictions.csv' generated successfully.")

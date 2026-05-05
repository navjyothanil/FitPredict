import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitPredict · AI Wellness Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #8888aa !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* Main background */
.main { background: #0a0a0f; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 40%, #ffcc02 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #666680;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #13131f, #1a1a28);
    border: 1px solid #222235;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.orange::before { background: linear-gradient(90deg, #ff6b35, #f7931e); }
.metric-card.teal::before   { background: linear-gradient(90deg, #00d4aa, #00b4d8); }
.metric-card.purple::before { background: linear-gradient(90deg, #a855f7, #ec4899); }
.metric-card.yellow::before { background: linear-gradient(90deg, #ffcc02, #f7931e); }
.metric-card.green::before  { background: linear-gradient(90deg, #22c55e, #16a34a); }

.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #55556a;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f0f0ff;
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #55556a;
    margin-top: 0.3rem;
    font-weight: 300;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e8f0;
    border-left: 3px solid #ff6b35;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-left: 4px solid #ff6b35;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c8c8d8;
}

/* Progress bar */
.progress-track {
    background: #1e1e2e;
    border-radius: 100px;
    height: 8px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.progress-fill {
    height: 8px;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff6b35, #ffcc02);
    transition: width 1s ease;
}

/* User info card */
.user-card {
    background: linear-gradient(145deg, #13131f, #1a1a28);
    border: 1px solid #222235;
    border-radius: 16px;
    padding: 1.5rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111118;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e1e2e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #55556a !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff6b35, #f7931e) !important;
    color: #fff !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #ff6b35, #f7931e) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Input fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #222235 !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
}

/* Tag badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-loss   { background: #1a2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-gain   { background: #1a1a2e; color: #60a5fa; border: 1px solid #1e40af; }
.badge-maint  { background: #2e2a1a; color: #fbbf24; border: 1px solid #92400e; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fitness_dashboard_dataset.csv")
    return df

@st.cache_resource
def train_models(df):
    le_gender   = LabelEncoder()
    le_activity = LabelEncoder()
    le_stress   = LabelEncoder()
    le_goal     = LabelEncoder()

    df2 = df.copy()
    df2['Gender_enc']   = le_gender.fit_transform(df2['Gender'])
    df2['Activity_enc'] = le_activity.fit_transform(df2['Activity_Level'])
    df2['Stress_enc']   = le_stress.fit_transform(df2['Stress_Level'])
    df2['Goal_enc']     = le_goal.fit_transform(df2['Goal_Type'])

    features = ['Age','Gender_enc','Height_cm','Current_Weight_kg','Target_Weight_kg',
                'Activity_enc','Goal_enc','Steps','Workout_Minutes','Sleep_Hours',
                'Water_Litres','Protein_g','Stress_enc','Calories_Burned']

    models, scores = {}, {}
    for target in ['Calories_Needed_kcal','Calories_to_Burn_kcal','Days_to_Goal']:
        X = df2[features]
        y = df2[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        m = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        scores[target] = {
            'mae': round(mean_absolute_error(y_te, y_pred), 1),
            'r2' : round(r2_score(y_te, y_pred), 3)
        }
        models[target] = m

    return models, scores, le_gender, le_activity, le_stress, le_goal, features

@st.cache_data
def compute_bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#60a5fa"
    elif bmi < 25: return "Normal",      "#4ade80"
    elif bmi < 30: return "Overweight",  "#fbbf24"
    else:          return "Obese",       "#f87171"

def predict(models, le_gender, le_activity, le_stress, le_goal, features, inp):
    row = pd.DataFrame([{
        'Age':               inp['age'],
        'Gender_enc':        le_gender.transform([inp['gender']])[0],
        'Height_cm':         inp['height'],
        'Current_Weight_kg': inp['current_weight'],
        'Target_Weight_kg':  inp['target_weight'],
        'Activity_enc':      le_activity.transform([inp['activity']])[0],
        'Goal_enc':          le_goal.transform([inp['goal']])[0],
        'Steps':             inp['steps'],
        'Workout_Minutes':   inp['workout'],
        'Sleep_Hours':       inp['sleep'],
        'Water_Litres':      inp['water'],
        'Protein_g':         inp['protein'],
        'Stress_enc':        le_stress.transform([inp['stress']])[0],
        'Calories_Burned':   inp['cal_burned'],
    }])
    return {t: max(0, round(models[t].predict(row[features])[0])) for t in models}

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#888899', size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor='#1e1e2e', zerolinecolor='#1e1e2e', tickfont=dict(color='#555566')),
    yaxis=dict(gridcolor='#1e1e2e', zerolinecolor='#1e1e2e', tickfont=dict(color='#555566')),
    colorway=['#ff6b35','#00d4aa','#a855f7','#ffcc02','#f87171','#60a5fa'],
)

# ─────────────────────────────────────────────
#  LOAD DATA + TRAIN
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Dataset not found. Make sure `fitness_dashboard_dataset.csv` is in the same folder.")
    st.stop()

models, scores, le_gender, le_activity, le_stress, le_goal, features = train_models(df)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
      <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(135deg,#ff6b35,#ffcc02);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        ⚡ FitPredict
      </div>
      <div style='font-size:0.7rem;color:#444455;letter-spacing:0.15em;text-transform:uppercase;margin-top:2px;'>
        AI Wellness Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode", ["🔍 Lookup User", "🧬 New Prediction"], label_visibility="collapsed")
    st.markdown("<hr style='border-color:#1e1e2e;margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    if "Lookup" in mode:
        st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#55556a;margin-bottom:0.4rem;'>Select User ID</div>", unsafe_allow_html=True)
        uid = st.selectbox("User", df['User_ID'].tolist(), label_visibility="collapsed")
        user_row = df[df['User_ID'] == uid].iloc[0]
        inp = {
            'age':            int(user_row['Age']),
            'gender':         user_row['Gender'],
            'height':         float(user_row['Height_cm']),
            'current_weight': float(user_row['Current_Weight_kg']),
            'target_weight':  float(user_row['Target_Weight_kg']),
            'activity':       user_row['Activity_Level'],
            'goal':           user_row['Goal_Type'],
            'steps':          int(user_row['Steps']),
            'workout':        int(user_row['Workout_Minutes']),
            'sleep':          float(user_row['Sleep_Hours']),
            'water':          float(user_row['Water_Litres']),
            'protein':        int(user_row['Protein_g']),
            'stress':         user_row['Stress_Level'],
            'cal_burned':     int(user_row['Calories_Burned']),
        }
    else:
        st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#55556a;margin-bottom:0.8rem;'>Personal Details</div>", unsafe_allow_html=True)
        age    = st.slider("Age",    18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.slider("Height (cm)", 140, 210, 170)
        c1, c2 = st.columns(2)
        with c1: cw = st.number_input("Current Wt (kg)", 40.0, 160.0, 75.0, step=0.5)
        with c2: tw = st.number_input("Target Wt (kg)",  40.0, 160.0, 65.0, step=0.5)

        st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#55556a;margin:1rem 0 0.5rem 0;'>Goal & Lifestyle</div>", unsafe_allow_html=True)
        goal     = st.selectbox("Goal Type",      ["Weight Loss","Weight Gain","Maintenance"])
        activity = st.selectbox("Activity Level", ["Sedentary","Lightly Active","Moderately Active","Very Active"])
        stress   = st.selectbox("Stress Level",   ["Low","Medium","High"])

        st.markdown("<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#55556a;margin:1rem 0 0.5rem 0;'>Daily Metrics</div>", unsafe_allow_html=True)
        steps     = st.slider("Steps",            1000,  20000, 8000, step=500)
        workout   = st.slider("Workout (min)",    0,     120,   45)
        sleep     = st.slider("Sleep (hrs)",      4.0,   10.0,  7.0, step=0.5)
        water     = st.slider("Water (litres)",   0.5,   5.0,   2.0, step=0.1)
        protein   = st.slider("Protein (g)",      30,    300,   120)
        cal_burned = st.slider("Calories Burned", 100,   1500,  500, step=50)

        inp = dict(age=age, gender=gender, height=height,
                   current_weight=cw, target_weight=tw,
                   activity=activity, goal=goal, stress=stress,
                   steps=steps, workout=workout, sleep=sleep,
                   water=water, protein=protein, cal_burned=cal_burned)

    st.markdown("<hr style='border-color:#1e1e2e;margin:1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    run = st.button("⚡  Run Prediction", use_container_width=True)

# ─────────────────────────────────────────────
#  COMPUTE PREDICTIONS (always show)
# ─────────────────────────────────────────────
preds = predict(models, le_gender, le_activity, le_stress, le_goal, features, inp)
cal_needed   = preds['Calories_Needed_kcal']
cal_to_burn  = preds['Calories_to_Burn_kcal']
days_to_goal = preds['Days_to_Goal']

bmi_val  = round(inp['current_weight'] / (inp['height']/100)**2, 1)
bmi_cat, bmi_color = compute_bmi_category(bmi_val)
wt_diff  = round(inp['current_weight'] - inp['target_weight'], 1)
bmr_val  = round(10*inp['current_weight'] + 6.25*inp['height'] - 5*inp['age'] + (5 if inp['gender']=='Male' else -161))

goal_badge = {
    'Weight Loss': '<span class="badge badge-loss">Weight Loss</span>',
    'Weight Gain': '<span class="badge badge-gain">Weight Gain</span>',
    'Maintenance': '<span class="badge badge-maint">Maintenance</span>',
}[inp['goal']]

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">FitPredict Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-Powered Calorie & Progress Intelligence</div>', unsafe_allow_html=True)
with col_h2:
    if "Lookup" in mode:
        st.markdown(f"""
        <div style='text-align:right;padding-top:0.5rem;'>
          <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8e8f0;'>{uid}</div>
          <div style='font-size:0.75rem;color:#55556a;'>{inp['gender']} · {inp['age']} yrs · {inp['height']} cm</div>
          <div style='margin-top:4px;'>{goal_badge}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e1e2e;margin:1rem 0 1.5rem 0;'>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Predictions", "📈  Analytics", "🤖  Model Info"])

# ════════════════════════════════════════════════
#  TAB 1 — PREDICTIONS
# ════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="metric-card orange">
          <div class="metric-label">Calories Needed</div>
          <div class="metric-value">{cal_needed:,}</div>
          <div class="metric-unit">kcal / day</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card teal">
          <div class="metric-label">Calories to Burn</div>
          <div class="metric-value">{cal_to_burn:,}</div>
          <div class="metric-unit">kcal / day</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        disp = f"{int(days_to_goal)}" if inp['goal'] != 'Maintenance' else "—"
        st.markdown(f"""
        <div class="metric-card purple">
          <div class="metric-label">Days to Goal</div>
          <div class="metric-value">{disp}</div>
          <div class="metric-unit">{'days' if inp['goal'] != 'Maintenance' else 'on track'}</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card yellow">
          <div class="metric-label">BMI</div>
          <div class="metric-value" style="color:{bmi_color};">{bmi_val}</div>
          <div class="metric-unit">{bmi_cat}</div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-label">BMR</div>
          <div class="metric-value">{bmr_val:,}</div>
          <div class="metric-unit">kcal (basal)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ─── Row 2: Gauge + Progress + Breakdown ──────────────────────────────
    col_g, col_p, col_b = st.columns([1.2, 1, 1])

    with col_g:
        st.markdown('<div class="section-header">Calorie Gauge</div>', unsafe_allow_html=True)
        act_mul = {'Sedentary':1.2,'Lightly Active':1.375,'Moderately Active':1.55,'Very Active':1.725}
        tdee_est = round(bmr_val * act_mul[inp['activity']])
        intake_pct = round(inp.get('cal_burned', cal_to_burn) / max(cal_needed, 1) * 100)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cal_needed,
            delta={'reference': tdee_est, 'font':{'size':14,'color':'#888899'}},
            gauge={
                'axis':{'range':[800, 4500], 'tickcolor':'#333344', 'tickfont':{'color':'#555566','size':10}},
                'bar':{'color':'#ff6b35','thickness':0.3},
                'bgcolor':'#13131f',
                'bordercolor':'#222235',
                'steps':[
                    {'range':[800,1500],'color':'#1a1a2e'},
                    {'range':[1500,2500],'color':'#151522'},
                    {'range':[2500,4500],'color':'#111118'},
                ],
                'threshold':{'line':{'color':'#ffcc02','width':3},'thickness':0.8,'value':tdee_est}
            },
            number={'font':{'family':'Syne','size':32,'color':'#ff6b35'},'suffix':' kcal'},
            title={'text':'Daily Intake Target','font':{'color':'#55556a','size':12}}
        ))
        fig_gauge.update_layout(**{**PLOT_LAYOUT, 'height':220, 'margin':dict(l=20,r=20,t=10,b=0)})
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar':False})

    with col_p:
        st.markdown('<div class="section-header">Weight Journey</div>', unsafe_allow_html=True)
        cw, tw2 = inp['current_weight'], inp['target_weight']
        if inp['goal'] == 'Maintenance':
            pct = 100
            label = "Maintaining"
        elif inp['goal'] == 'Weight Loss':
            pct = max(0, min(100, round((1 - (cw - tw2) / max(cw, 1)) * 100))) if cw > tw2 else 100
            label = f"{abs(wt_diff)} kg to lose"
        else:
            pct = max(0, min(100, round((cw / tw2) * 100))) if tw2 > cw else 100
            label = f"{abs(wt_diff)} kg to gain"

        st.markdown(f"""
        <div class="user-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:1rem;">
            <div>
              <div style="font-size:0.7rem;color:#55556a;letter-spacing:0.1em;text-transform:uppercase;">Current</div>
              <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#ff6b35;">{cw} kg</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:0.7rem;color:#55556a;letter-spacing:0.1em;text-transform:uppercase;">Target</div>
              <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#00d4aa;">{tw2} kg</div>
            </div>
          </div>
          <div style="font-size:0.75rem;color:#55556a;margin-bottom:0.4rem;">{label}</div>
          <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;"></div>
          </div>
          <div style="font-size:0.7rem;color:#444455;margin-top:0.4rem;text-align:right;">{pct}% {'complete' if inp['goal']!='Maintenance' else ''}</div>
          <hr style="border-color:#1e1e2e;margin:1rem 0;">
          <div style="display:flex;justify-content:space-between;">
            <div style="text-align:center;">
              <div style="font-size:0.65rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;">Activity</div>
              <div style="font-size:0.8rem;color:#c8c8d8;font-weight:500;margin-top:2px;">{inp['activity'].replace(' ',chr(10))}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:0.65rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;">Stress</div>
              <div style="font-size:0.85rem;color:#c8c8d8;font-weight:600;margin-top:2px;">{inp['stress']}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:0.65rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;">Sleep</div>
              <div style="font-size:0.85rem;color:#c8c8d8;font-weight:600;margin-top:2px;">{inp['sleep']}h</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Energy Breakdown</div>', unsafe_allow_html=True)
        labels = ['Basal (BMR)','Activity','Workout','Steps']
        step_cal = round(inp['steps'] * 0.04)
        met_m = {'Sedentary':3,'Lightly Active':4.5,'Moderately Active':6,'Very Active':8}
        wk_cal = round(met_m[inp['activity']] * inp['current_weight'] * (inp['workout']/60))
        activity_cal = tdee_est - bmr_val - wk_cal - step_cal
        values = [max(bmr_val,0), max(activity_cal,0), wk_cal, step_cal]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.6,
            marker=dict(colors=['#ff6b35','#f7931e','#00d4aa','#a855f7'],
                        line=dict(color='#0a0a0f', width=2)),
            textfont=dict(size=10, color='#c8c8d8'),
            hovertemplate='<b>%{label}</b><br>%{value} kcal<extra></extra>'
        ))
        fig_pie.add_annotation(
            text=f"<b>{tdee_est}</b><br><span style='font-size:10px'>TDEE</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family='Syne', size=18, color='#e8e8f0'), align='center'
        )
        fig_pie.update_layout(**{**PLOT_LAYOUT, 'height':240, 'showlegend':True,
            'legend':dict(orientation='v', x=1.02, y=0.5, font=dict(size=10,color='#888899'))})
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar':False})

    # ─── Insights ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 AI Insights</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)

    daily_deficit = abs(tdee_est - cal_needed)
    weeks = round(days_to_goal / 7, 1) if inp['goal'] != 'Maintenance' else 0

    with ic1:
        if inp['goal'] == 'Weight Loss':
            msg = f"With a daily deficit of ~{daily_deficit} kcal, you're targeting {weeks} weeks to reach {inp['target_weight']} kg. Consistent workout and sleep will accelerate this."
        elif inp['goal'] == 'Weight Gain':
            msg = f"A surplus of ~{daily_deficit} kcal per day supports muscle gain. Aim for {inp['protein']}g+ protein and resistance training to maximize lean mass."
        else:
            msg = f"You're in maintenance mode. Your TDEE of {tdee_est} kcal keeps you at {cw} kg. Focus on performance, sleep quality, and recovery."
        st.markdown(f'<div class="insight-box">🎯 <b>Goal Insight</b><br>{msg}</div>', unsafe_allow_html=True)

    with ic2:
        sleep_msg = "excellent" if inp['sleep'] >= 7.5 else "adequate" if inp['sleep'] >= 6 else "insufficient"
        water_msg = "well hydrated" if inp['water'] >= 2.5 else "slightly under target" if inp['water'] >= 1.8 else "under-hydrated"
        st.markdown(f'<div class="insight-box">😴 <b>Recovery Status</b><br>Sleep is <b>{sleep_msg}</b> at {inp["sleep"]}h. Hydration is <b>{water_msg}</b> at {inp["water"]}L. {'Increase water intake by 0.5L for optimal metabolism.' if inp["water"] < 2.0 else "Keep it up!"}</div>', unsafe_allow_html=True)

    with ic3:
        steps_feedback = "great NEAT activity" if inp['steps'] >= 10000 else "moderate NEAT" if inp['steps'] >= 6000 else "low NEAT — try increasing daily steps"
        prot_feedback  = "optimal" if inp['protein'] >= inp['current_weight']*1.6 else "slightly low — aim for 1.6–2.0g/kg"
        st.markdown(f'<div class="insight-box">🏃 <b>Activity & Nutrition</b><br>{inp["steps"]:,} steps = <b>{steps_feedback}</b>. Protein at {inp["protein"]}g is <b>{prot_feedback}</b> for your body weight.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)

    with a1:
        # Calories Needed by Goal Type — violin
        fig_v = go.Figure()
        colors = {'Weight Loss':'#ff6b35','Weight Gain':'#60a5fa','Maintenance':'#fbbf24'}
        for g, c in colors.items():
            sub = df[df['Goal_Type']==g]['Calories_Needed_kcal']
            fig_v.add_trace(go.Violin(y=sub, name=g, fillcolor=c, line_color=c,
                                      opacity=0.7, box_visible=True,
                                      meanline_visible=True,
                                      meanline_color='#ffffff'))
        fig_v.update_layout(**{**PLOT_LAYOUT, 'title':'Calories Needed by Goal Type',
                               'height':320, 'showlegend':False,
                               'yaxis':dict(title='kcal/day', gridcolor='#1e1e2e')})
        st.plotly_chart(fig_v, use_container_width=True, config={'displayModeBar':False})

    with a2:
        # Steps vs Calories Burned scatter
        sample = df.sample(300, random_state=1)
        fig_s = px.scatter(sample, x='Steps', y='Calories_Burned', color='Activity_Level',
                           size='Workout_Minutes', opacity=0.7,
                           color_discrete_map={'Sedentary':'#f87171','Lightly Active':'#fbbf24',
                                               'Moderately Active':'#4ade80','Very Active':'#60a5fa'})
        fig_s.update_layout(**{**PLOT_LAYOUT, 'title':'Steps vs Calories Burned',
                               'height':320, 'legend':dict(font=dict(size=9,color='#888899'))})
        st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar':False})

    a3, a4 = st.columns(2)

    with a3:
        # BMI Distribution histogram
        fig_h = go.Figure()
        for g, c in [('Male','#60a5fa'),('Female','#f472b6')]:
            fig_h.add_trace(go.Histogram(x=df[df['Gender']==g]['BMI'], name=g,
                                         marker_color=c, opacity=0.65,
                                         xbins=dict(size=1)))
        fig_h.update_layout(**{**PLOT_LAYOUT, 'title':'BMI Distribution by Gender',
                               'barmode':'overlay', 'height':300,
                               'xaxis':dict(title='BMI', gridcolor='#1e1e2e'),
                               'yaxis':dict(title='Count', gridcolor='#1e1e2e'),
                               'legend':dict(font=dict(size=10,color='#888899'))})
        st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar':False})

    with a4:
        # Days to Goal by Activity Level box
        fig_b = go.Figure()
        act_colors = {'Sedentary':'#f87171','Lightly Active':'#fbbf24',
                      'Moderately Active':'#4ade80','Very Active':'#60a5fa'}
        sub_df = df[df['Goal_Type']!='Maintenance']
        for act, c in act_colors.items():
            d = sub_df[sub_df['Activity_Level']==act]['Days_to_Goal']
            fig_b.add_trace(go.Box(y=d, name=act.replace(' ','<br>'),
                                   marker_color=c, line_color=c, fillcolor=c,
                                   opacity=0.7, boxmean='sd'))
        fig_b.update_layout(**{**PLOT_LAYOUT, 'title':'Days to Goal by Activity Level',
                               'height':300, 'showlegend':False,
                               'yaxis':dict(title='Days', gridcolor='#1e1e2e')})
        st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar':False})

    # ─── Weekly Simulated Progress ──────────────────────────────────────────
    st.markdown('<div class="section-header">Simulated Weight Progress (Your Plan)</div>', unsafe_allow_html=True)

    if inp['goal'] != 'Maintenance' and days_to_goal > 0:
        total_weeks = min(int(days_to_goal // 7) + 1, 52)
        weeks_range = list(range(0, total_weeks + 1))
        wt_per_week = abs(wt_diff) / max(total_weeks, 1)

        sim_weight = []
        for w in weeks_range:
            if inp['goal'] == 'Weight Loss':
                sim_weight.append(round(inp['current_weight'] - wt_per_week * w + np.random.normal(0, 0.2), 1))
            else:
                sim_weight.append(round(inp['current_weight'] + wt_per_week * w + np.random.normal(0, 0.2), 1))

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(
            x=weeks_range, y=sim_weight,
            mode='lines+markers',
            line=dict(color='#ff6b35', width=2.5),
            marker=dict(size=4, color='#ff6b35'),
            fill='tozeroy',
            fillcolor='rgba(255,107,53,0.08)',
            name='Projected Weight'
        ))
        fig_prog.add_hline(y=inp['target_weight'],
                           line_dash='dash', line_color='#00d4aa', line_width=1.5,
                           annotation_text=f"Target {inp['target_weight']} kg",
                           annotation_font_color='#00d4aa', annotation_font_size=11)
        fig_prog.update_layout(**{**PLOT_LAYOUT,
            'height':280, 'showlegend':False,
            'xaxis':dict(title='Week', gridcolor='#1e1e2e'),
            'yaxis':dict(title='Weight (kg)', gridcolor='#1e1e2e')})
        st.plotly_chart(fig_prog, use_container_width=True, config={'displayModeBar':False})
    else:
        st.markdown('<div class="insight-box">✅ You are in Maintenance mode — no weight change projected. Focus on consistency!</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  TAB 3 — MODEL INFO
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    target_labels = {
        'Calories_Needed_kcal':  ('Calories Needed', '🟠'),
        'Calories_to_Burn_kcal': ('Calories to Burn','🔵'),
        'Days_to_Goal':          ('Days to Goal',    '🟣'),
    }
    for col, (target, (label, icon)) in zip([m1,m2,m3], target_labels.items()):
        with col:
            r2  = scores[target]['r2']
            mae = scores[target]['mae']
            st.markdown(f"""
            <div class="user-card" style="text-align:center;">
              <div style="font-size:1.5rem;margin-bottom:0.5rem;">{icon}</div>
              <div style="font-family:Syne,sans-serif;font-size:0.85rem;font-weight:700;color:#e8e8f0;margin-bottom:1rem;">{label}</div>
              <div style="display:flex;justify-content:space-around;">
                <div>
                  <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                              color:{'#4ade80' if r2>0.85 else '#fbbf24' if r2>0.7 else '#f87171'};">{r2}</div>
                  <div style="font-size:0.65rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;">R² Score</div>
                </div>
                <div>
                  <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#60a5fa;">±{mae}</div>
                  <div style="font-size:0.65rem;color:#55556a;text-transform:uppercase;letter-spacing:0.1em;">MAE</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    fi_col1, fi_col2 = st.columns(2)

    feat_display = ['Age','Gender','Height','Curr. Weight','Target Weight',
                    'Activity','Goal Type','Steps','Workout Min',
                    'Sleep Hrs','Water','Protein','Stress','Cal. Burned']

    for col, target in zip([fi_col1, fi_col2], ['Calories_Needed_kcal','Days_to_Goal']):
        with col:
            importances = models[target].feature_importances_
            sorted_idx  = np.argsort(importances)[-10:]
            fig_fi = go.Figure(go.Bar(
                x=importances[sorted_idx],
                y=[feat_display[i] for i in sorted_idx],
                orientation='h',
                marker=dict(
                    color=importances[sorted_idx],
                    colorscale=[[0,'#1a1a2e'],[0.5,'#f7931e'],[1,'#ff6b35']],
                    showscale=False
                )
            ))
            fig_fi.update_layout(**{**PLOT_LAYOUT,
                'title': f'Top Features: {target_labels[target][0]}',
                'height':300,
                'xaxis':dict(title='Importance', gridcolor='#1e1e2e'),
                'yaxis':dict(gridcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar':False})

    st.markdown("""
    <div class="insight-box">
    🤖 <b>Algorithm:</b> Gradient Boosting Regressor (200 estimators, depth=5, lr=0.08) · 
    Trained on 80% of 1,500 synthetic records · Evaluated on held-out 20% test set · 
    Features: Age, Gender, Height, Weights, Activity, Goal, Steps, Workout, Sleep, Water, Protein, Stress, Calories Burned
    </div>
    """, unsafe_allow_html=True)

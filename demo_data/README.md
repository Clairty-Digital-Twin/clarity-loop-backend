# Clarity Digital Twin Demo Data

**Generated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

This directory contains comprehensive synthetic data for demonstrating the Clarity Digital Twin platform to potential cofounders and stakeholders.

## 🎯 **Demo Scenarios**

### 1. **Apple HealthKit Integration Demo**
**Location:** `healthkit/`  
**Purpose:** Showcase mature HealthKit data ingestion and natural language chat interface

**Key Components:**
- `users.json` - 3 synthetic user profiles with varied demographics
- `health_metrics.json` - 63 realistic health data points across 7 days
- `activities.json` - 49 activity records (walking, running, sleep, etc.)
- `predictions.json` - 9 PAT model predictions and analysis results
- `reports.json` - 3 comprehensive health analysis reports

**Demo Flow:**
1. Load user profile: "Meet Sarah, our health-conscious user..."
2. Show rich HealthKit data: Sleep patterns, activity levels, heart rate
3. Demonstrate chat interface: "What can you tell me about my sleep quality this week?"
4. Show AI-powered insights and personalized recommendations

### 2. **Bipolar Risk Detection System**
**Location:** `clinical/`  
**Purpose:** Early warning system for mood episodes with clinical-grade accuracy

**Key Components:**
- `baseline_scenario.json` - 30 days of normal baseline patterns
- `prodromal_scenario.json` - 14 days of early warning signs
- `acute_manic_scenario.json` - 7 days of acute episode data
- `recovery_scenario.json` - 21 days of recovery phase
- `complete_episode_scenario.json` - Full 72-day episode timeline

**Demo Flow:**
1. Show baseline: "Here's what normal looks like for this patient..."
2. Highlight prodromal phase: "Notice the sleep disruption starting 2 weeks before..."
3. Show acute phase: "Risk score jumps to 0.9 - clinical intervention needed"
4. Recovery tracking: "Monitoring return to baseline patterns"

### 3. **Multi-Modal Chat Interface**
**Location:** `chat/`  
**Purpose:** Natural language interaction with personal health data

**Personas Available:**
- `health_conscious/` - Fitness enthusiast asking optimization questions
- `clinical_patient/` - Patient seeking guidance on mood patterns

**Components per Persona:**
- `conversation_starters.json` - Natural opening questions
- `followup_questions.json` - Contextual follow-ups
- `response_templates.json` - AI response patterns
- `demo_conversations.json` - Complete conversation examples

---

## 🚀 **Quick Demo Setup**

### Upload to S3 for Matt's Demo
```bash
# Upload all demo data to S3
aws s3 sync demo_data/ s3://clarity-demo-data/ --delete

# Verify upload
aws s3 ls s3://clarity-demo-data/ --recursive
```

### Validate Data Quality
```bash
# Check data integrity
python3 scripts/validate_demo_data.py --path demo_data/

# Test PAT model with synthetic data
python3 scripts/test_pat_integration.py --data demo_data/healthkit/activities.json
```

### Run Local Demo
```bash
# Start backend with demo data
make dev

# Load demo context
curl -X POST localhost:8000/api/v1/demo/load \
  -H "Content-Type: application/json" \
  -d '{"scenario": "healthkit_chat", "user_id": "demo-user-1"}'
```

---

## 📊 **Data Specifications**

### HealthKit Data Format
```json
{
  "user_id": "USER#...",
  "timestamp": "2024-01-15T08:30:00Z",
  "metric_type": "heart_rate",
  "value": 72.5,
  "unit": "bpm",
  "metadata": {
    "device": "Apple Watch Series 9",
    "quality": "high"
  }
}
```

### Clinical Risk Format
```json
{
  "date": "2024-01-15T00:00:00",
  "phase": "prodromal",
  "risk_score": 0.45,
  "risk_level": "moderate",
  "sleep_hours": 5.2,
  "mood_score": 4.1,
  "clinical_notes": "Sleep disruption pattern emerging"
}
```

### Chat Context Format
```json
{
  "intent": "sleep_quality_inquiry",
  "user_message": "How was my sleep this week?",
  "context_needed": ["sleep_metrics", "activity_data"],
  "response_template": "Based on your data...",
  "followup_suggestions": ["What affects your sleep?", "Want tips?"]
}
```

---

## 🔒 **Privacy & Security**

- ✅ **Synthetic Data Only** - No real user information
- ✅ **HIPAA Compliant** - No PII or PHI included
- ✅ **Anonymized IDs** - UUID-based identifiers
- ✅ **Clinical Validity** - Realistic but fictional patterns
- ✅ **Time-shifted** - Data appears recent but is synthetic

---

## 🎬 **Demo Script Suggestions**

### Opening (30 seconds)
> "Let me show you our two core innovations that make Clarity unique in the digital health space..."

### HealthKit Demo (2 minutes)
> "First, our mature Apple HealthKit integration. Unlike other platforms that just collect data, we make it conversational..."
> 
> *[Show natural chat with health data]*
> 
> "Ask me anything about your health patterns, and get insights in plain English."

### Risk Detection Demo (2 minutes)
> "Second, our breakthrough in bipolar risk detection. We can identify mood episodes up to 2 weeks before they occur..."
> 
> *[Show risk progression timeline]*
> 
> "This isn't just data visualization - this is clinical-grade early warning that saves lives."

### Technical Architecture (1 minute)
> "Behind the scenes: AWS ECS deployment, PAT transformer models, real-time inference, and multi-modal data fusion..."

---

## 📈 **Success Metrics**

**Technical Demonstration:**
- [ ] HealthKit data loads and displays correctly
- [ ] Chat interface responds naturally to health queries
- [ ] Risk detection timeline shows clear progression
- [ ] PAT model predictions are realistic and explainable

**Business Impact:**
- [ ] Demonstrates clear value proposition vs competitors
- [ ] Shows technical sophistication and clinical applicability
- [ ] Proves platform readiness for real users
- [ ] Illustrates scalability and AWS integration

---

*Ready to revolutionize digital health? Let's build the future together.* 🚀 
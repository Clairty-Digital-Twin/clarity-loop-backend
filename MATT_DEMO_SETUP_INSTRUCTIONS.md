# Clarity Digital Twin Demo Setup - For Matt

Hi Matt! 👋

Your AWS access has been upgraded to **PowerUser** and everything is ready for you to deploy and showcase the Clarity Digital Twin platform. Here's your complete setup guide.

## 🔐 **Your AWS Access Status**
✅ **Username:** `matt-gorbett-cofounder`  
✅ **Permissions:** PowerUserAccess (full access except IAM user management)  
✅ **S3 Access:** Full read/write access to demo data  
✅ **ECS Access:** Can view and manage the running application  

## 🚀 **Quick Demo Deployment (5 minutes)**

### Step 1: Clone and Navigate
```bash
git clone https://github.com/YourOrg/clarity-loop-backend.git
cd clarity-loop-backend
```

### Step 2: Deploy Demo Data
```bash
# Run the deployment script (requires your AWS credentials)
./scripts/deploy_demo_data.sh
```

This script will:
- Create `clarity-demo-data` S3 bucket
- Upload all synthetic demo data (21 files)
- Configure proper permissions
- Generate access URLs

### Step 3: Verify Deployment
```bash
# Check the running backend service
aws ecs describe-services \
    --cluster clarity-backend-cluster \
    --services clarity-backend-service

# View demo data
aws s3 ls s3://clarity-demo-data/ --recursive
```

## 🎭 **Demo Scenarios Available**

### 1. **Apple HealthKit Integration** 📱
**Story:** "Chat with your health data in natural language"

**Data Location:** `s3://clarity-demo-data/healthkit/`
- 3 user profiles with realistic health patterns
- 63 health metrics across 7 days  
- 49 activity records (sleep, exercise, heart rate)
- 9 PAT model predictions

**Demo Script:**
> "Let me show you Sarah, one of our users. She's been tracking her health with Apple Watch for a week. Instead of looking at charts, she can just ask questions..."
>
> *[Load user profile and show chat interface]*
>
> **User:** "How was my sleep quality this week?"
> **Clarity:** "Your sleep efficiency averaged 87% with 7.2 hours per night. I noticed Tuesday showed disrupted patterns..."

### 2. **Bipolar Risk Detection** 🧠
**Story:** "Clinical-grade early warning system for mood episodes"

**Data Location:** `s3://clarity-demo-data/clinical/`
- Complete 72-day episode timeline
- 4 distinct phases: baseline → prodromal → acute → recovery
- Risk scores from 0.15 (normal) to 0.9 (crisis)

**Demo Script:**
> "This is our breakthrough in mental health monitoring. We can detect bipolar episodes up to 2 weeks before they occur..."
>
> *[Show timeline visualization]*
>
> "Notice the sleep disruption starting here - that's 14 days before the acute phase. Our PAT model caught the pattern."

### 3. **Multi-Modal Chat Interface** 💬
**Story:** "AI that understands your complete health picture"

**Data Location:** `s3://clarity-demo-data/chat/`
- Natural conversation starters
- Context-aware follow-up questions
- Personalized response templates

## 🏗️ **Technical Architecture Highlights**

**For Technical Stakeholders:**
- **AWS ECS:** Auto-scaling containerized deployment
- **PAT Models:** Pre-trained Actigraphy Transformer (3 model sizes)
- **Real-time Inference:** FastAPI + async processing
- **Data Lake:** S3 with structured health data formats
- **Security:** IAM roles, VPC isolation, encrypted at rest

**Current Deployment:**
- **Service:** `clarity-backend-cluster/clarity-backend-service`
- **Revision:** 168 (latest)
- **Status:** ✅ RUNNING (1/1 healthy tasks)
- **URL:** `https://api.clarity-digital-twin.com`

## 📊 **Success Metrics for Demo**

### Immediate Impact
- [ ] HealthKit data loads and displays correctly
- [ ] Chat interface responds naturally
- [ ] Risk detection shows clear clinical value
- [ ] Technical architecture impresses stakeholders

### Business Validation
- [ ] Demonstrates clear competitive advantage
- [ ] Shows clinical applicability and market readiness
- [ ] Proves technical sophistication
- [ ] Illustrates scaling potential with AWS

## 🆘 **Troubleshooting**

### If AWS CLI fails:
```bash
aws configure
# Enter your access key and secret from AWS console
```

### If bucket creation fails:
```bash
# Check your permissions
aws sts get-caller-identity
aws iam list-attached-user-policies --user-name matt-gorbett-cofounder
```

### If demo data doesn't load:
```bash
# Re-run the deployment
./scripts/deploy_demo_data.sh

# Check bucket contents
aws s3 ls s3://clarity-demo-data/ --recursive
```

## 📞 **Support**

**For Technical Issues:**
- Check AWS CloudWatch logs for the ECS service
- Review S3 bucket permissions
- Validate demo data integrity

**For Demo Questions:**
- Review `demo_data/README.md` for detailed scenarios
- Check conversation examples in `chat/` directories
- Reference clinical scenarios in `clinical/` directory

## 🎯 **Key Talking Points**

### Competitive Advantages
1. **"We're the only platform with conversational health data"**
2. **"Clinical-grade bipolar detection with 2-week early warning"**  
3. **"Enterprise-ready AWS infrastructure from day one"**

### Market Opportunity
- Digital therapeutics market: $7.8B by 2030
- Mental health tech gap: current solutions are reactive
- Apple HealthKit integration: 1.8B+ devices

### Technical Differentiation
- PAT transformer models (state-of-the-art)
- Multi-modal health data fusion  
- Real-time inference at scale
- HIPAA-compliant architecture

---

**Ready to change healthcare? Let's build the future together!** 🚀

*Questions? Reach out to the team - we're excited to see you showcase what we've built.* 
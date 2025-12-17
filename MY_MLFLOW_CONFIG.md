# MLflow Remote Setup - Your Configuration

## Your Credentials (SAVE THIS FILE!)

### Neon PostgreSQL
```
Connection String:
postgresql://neondb_owner:npg_xCgJ7tFQhP5G@ep-steep-mud-ahjizix4-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require
```

### AWS S3
```
Bucket Name: mlops-airquality-artifacts-yashwanth
Region: us-east-1
```

### AWS Credentials
```
Access Key ID: <YOUR_AWS_ACCESS_KEY_ID>
Secret Access Key: <YOUR_AWS_SECRET_ACCESS_KEY>
```

### EC2 Instance
```
Public IP: 44.220.133.154
Key Pair: mlflow-key.pem (in Downloads folder)
SSH Command: ssh -i mlflow-key.pem ubuntu@44.220.133.154
```

---

## Commands for EC2 Setup

### On EC2 Instance (after SSH):

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow
pip install mlflow boto3 psycopg2-binary

# Configure AWS credentials
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = <YOUR_AWS_ACCESS_KEY_ID>
aws_secret_access_key = <YOUR_AWS_SECRET_ACCESS_KEY>
region = us-east-1
EOF

# Start MLflow server
mlflow server \
  --backend-store-uri "postgresql://neondb_owner:npg_xCgJ7tFQhP5G@ep-steep-mud-ahjizix4-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require" \
  --default-artifact-root "s3://mlops-airquality-artifacts-yashwanth" \
  --host 0.0.0.0 \
  --port 5000
```

---

## Local .env Configuration

Create/update `.env` file in your project:

```ini
MLFLOW_TRACKING_URI=http://44.220.133.154:5000
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=us-east-1
```

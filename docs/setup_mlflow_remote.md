# Remote MLflow Setup Guide

This guide walks you through setting up a remote MLflow tracking server on AWS with PostgreSQL backend and S3 artifact storage.

---

## Prerequisites

- AWS Account
- AWS CLI configured locally
- Basic knowledge of EC2, S3, and PostgreSQL

---

## Step 1: PostgreSQL Backend Setup (Neon.com)

### 1.1 Create Free PostgreSQL Database

1. Go to [Neon.com](https://neon.com/pricing)
2. Sign up for a free account
3. Create a new project: `mlflow-tracking`
4. Create a database: `mlflow_db`
5. Copy the connection string:
   ```
   postgresql://username:password@host:5432/mlflow_db
   ```

### 1.2 Test Connection

```powershell
# Install psycopg2 if not already installed
pip install psycopg2-binary

# Test connection
python -c "import psycopg2; conn = psycopg2.connect('YOUR_CONNECTION_STRING'); print('✓ Connected')"
```

---

## Step 2: S3 Artifact Store Setup

### 2.1 Create S3 Bucket

```bash
# Create bucket (replace with your bucket name)
aws s3 mb s3://mlops-airquality-artifacts --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket mlops-airquality-artifacts \
  --versioning-configuration Status=Enabled
```

### 2.2 Create IAM User for MLflow

1. Go to AWS IAM Console
2. Create new user: `mlflow-user`
3. Attach policy: `AmazonS3FullAccess` (or create custom policy)
4. Generate access keys
5. Save `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

**Custom S3 Policy (more secure):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::mlops-airquality-artifacts",
        "arn:aws:s3:::mlops-airquality-artifacts/*"
      ]
    }
  ]
}
```

---

## Step 3: EC2 Instance Setup

### 3.1 Launch EC2 Instance

1. Go to EC2 Console
2. Launch instance:
   - **AMI:** Ubuntu Server 22.04 LTS
   - **Instance Type:** t2.micro (free tier eligible)
   - **Key Pair:** Create or use existing
   - **Security Group:** Create new with rules:
     - SSH (22) from your IP
     - Custom TCP (5000) from anywhere (or your IP)
3. Launch instance

### 3.2 Connect to Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 3.3 Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install PostgreSQL client
sudo apt install postgresql-client -y

# Create virtual environment
python3 -m venv mlflow-env
source mlflow-env/bin/activate

# Install MLflow and dependencies
pip install mlflow boto3 psycopg2-binary
```

### 3.4 Configure AWS Credentials

```bash
# Create AWS credentials file
mkdir -p ~/.aws
nano ~/.aws/credentials
```

Add:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-1
```

### 3.5 Start MLflow Server

```bash
# Set environment variables
export BACKEND_STORE_URI="postgresql://username:password@host:5432/mlflow_db"
export ARTIFACT_ROOT="s3://mlops-airquality-artifacts"

# Start MLflow server
mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
```

### 3.6 Create Systemd Service (Optional - for auto-start)

```bash
sudo nano /etc/systemd/system/mlflow.service
```

Add:
```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="BACKEND_STORE_URI=postgresql://username:password@host:5432/mlflow_db"
Environment="ARTIFACT_ROOT=s3://mlops-airquality-artifacts"
Environment="AWS_ACCESS_KEY_ID=YOUR_KEY"
Environment="AWS_SECRET_ACCESS_KEY=YOUR_SECRET"
ExecStart=/home/ubuntu/mlflow-env/bin/mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

---

## Step 4: Local Configuration

### 4.1 Update .env File

```bash
# In your local project directory
cp .env.example .env
nano .env
```

Update:
```ini
MLFLOW_TRACKING_URI=http://<EC2_PUBLIC_IP>:5000
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
AWS_DEFAULT_REGION=us-east-1
MLFLOW_S3_ENDPOINT_URL=
```

### 4.2 Test Connection

```python
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")

# Test connection
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
print(f"✓ Connected! Found {len(experiments)} experiments")
```

---

## Step 5: Verification

### 5.1 Access MLflow UI

Open browser: `http://<EC2_PUBLIC_IP>:5000`

### 5.2 Run Test Training

```powershell
# In your local project
uv run python scripts/05_train_xgboost.py
```

Check MLflow UI to verify:
- Experiment appears
- Run is logged
- Metrics are recorded
- Artifacts are in S3

---

## Troubleshooting

### Connection Refused
- Check EC2 security group allows port 5000
- Verify MLflow server is running: `sudo systemctl status mlflow`

### S3 Permission Denied
- Verify IAM user has S3 permissions
- Check AWS credentials are correct

### PostgreSQL Connection Error
- Test connection string from EC2 instance
- Ensure Neon.com database is accessible

### Artifacts Not Uploading
- Verify S3 bucket exists
- Check AWS credentials on EC2 instance

---

## Cost Optimization

- **EC2:** Use t2.micro (free tier) or stop when not in use
- **S3:** Enable lifecycle policies to archive old artifacts
- **PostgreSQL:** Neon.com free tier is sufficient for this project

---

## Security Best Practices

1. **Restrict Security Groups:** Only allow your IP for SSH and MLflow
2. **Use IAM Roles:** Instead of access keys on EC2 (more secure)
3. **Enable S3 Encryption:** Server-side encryption for artifacts
4. **Rotate Credentials:** Regularly update AWS access keys
5. **Use VPC:** For production, deploy in private VPC

---

## Next Steps

After setup:
1. Re-run all training scripts to log to remote server
2. Take screenshots of MLflow UI for documentation
3. Update README with remote tracking URI
4. Test API with models from remote registry

---

## Support

For issues:
- AWS Documentation: https://docs.aws.amazon.com/
- MLflow Documentation: https://mlflow.org/docs/latest/
- Neon.com Support: https://neon.tech/docs/

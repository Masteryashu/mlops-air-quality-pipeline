# Commands to Run in EC2 Terminal

Copy and paste these blocks one by one into your browser terminal.

## 1. Configure AWS Credentials
(Copy this entire block and paste it)

```bash
mkdir -p ~/.aws

cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = <YOUR_AWS_ACCESS_KEY_ID>
aws_secret_access_key = <YOUR_AWS_SECRET_ACCESS_KEY>
region = us-east-1
EOF

echo "✅ AWS Credentials configured"
```

## 2. Start MLflow Server
(Copy this command - it's one long line)

```bash
/home/ubuntu/mlflow-env/bin/mlflow server \
  --backend-store-uri "postgresql://neondb_owner:npg_xCgJ7tFQhP5G@ep-steep-mud-ahjizix4-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require" \
  --default-artifact-root "s3://mlops-airquality-artifacts-yashwanth" \
  --host 0.0.0.0 \
  --port 5000
```

---

## 3. Verification

After you run step 2, you should see output saying:
`[INFO] Listening at: http://0.0.0.0:5000`

Then, open this URL in your web browser:
**http://44.220.133.154:5000**

You should see the MLflow UI!

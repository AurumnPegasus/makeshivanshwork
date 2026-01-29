#!/bin/bash
# Deploy to Google Cloud Run
# Prerequisites: gcloud CLI installed and authenticated

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
SERVICE_NAME="makearjowork"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "=== Building Docker image ==="
docker build -t $IMAGE_NAME .

echo "=== Pushing to Google Container Registry ==="
docker push $IMAGE_NAME

echo "=== Deploying to Cloud Run ==="
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "DOMAIN=https://$SERVICE_NAME-$PROJECT_ID.run.app" \
    --set-secrets "SECRET_KEY=secret-key:latest,SMTP_USER=smtp-user:latest,SMTP_PASS=smtp-pass:latest,FROM_EMAIL=from-email:latest"

echo ""
echo "=== Deployment complete! ==="
echo "Your service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

echo ""
echo "=== Next steps ==="
echo "1. Set up secrets in Google Secret Manager:"
echo "   gcloud secrets create secret-key --data-file=-"
echo "   gcloud secrets create smtp-user --data-file=-"
echo "   gcloud secrets create smtp-pass --data-file=-"
echo ""
echo "2. Map your custom domain:"
echo "   gcloud beta run domain-mappings create --service $SERVICE_NAME --domain makearjowork.com --region $REGION"

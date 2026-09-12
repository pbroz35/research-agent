#!/usr/bin/env bash
# Deploy the MCP server to Cloud Run.
#
#   PROJECT_ID=my-project ./deploy.sh
#
# First run also creates the MCP_AUTH_TOKEN secret and grants the runtime
# service account read access to it. Re-runs are idempotent.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mcp-server}"
SECRET_NAME="${SECRET_NAME:-mcp-auth-token}"
# Non-secret configuration; overridable per deploy.
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-https://openrouter.ai/api/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-openai/text-embedding-3-small}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "PROJECT_ID is not set. Run: gcloud config set project <id>" >&2
  exit 1
fi

echo "==> project=${PROJECT_ID} region=${REGION} service=${SERVICE}"

# Project IDs are globally unique, so a plausible-looking name may well be
# someone else's project. Fail here with something readable rather than letting
# every later call return AUTH_PERMISSION_DENIED.
if ! gcloud projects describe "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Cannot access project '${PROJECT_ID}' — it either does not exist or belongs" >&2
  echo "to someone else. Projects you can see:" >&2
  gcloud projects list --format='value(projectId)' | sed 's/^/  /' >&2
  echo "Create one with: gcloud projects create <globally-unique-id>" >&2
  exit 1
fi

# compute.googleapis.com is in the list because enabling it provisions the
# default compute service account, which is the Cloud Run runtime identity the
# secret binding below is granted to. On a brand-new project that account does
# not exist yet, and the binding fails.
echo "==> Ensuring required APIs are enabled"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  compute.googleapis.com \
  --project="${PROJECT_ID}"

if ! gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "==> Creating secret ${SECRET_NAME} with a fresh random token"
  gcloud secrets create "${SECRET_NAME}" --replication-policy=automatic --project="${PROJECT_ID}"
  # tr -d '\n': a trailing newline in the payload survives into the
  # container's env var but is stripped by any client using $(...).
  openssl rand -hex 32 | tr -d '\n' | gcloud secrets versions add "${SECRET_NAME}" \
    --data-file=- --project="${PROJECT_ID}"
fi

# Seed a secret from the matching key in the local .env, if it is not already in
# Secret Manager. Existing secrets are left alone; rotate them with
# `gcloud secrets versions add` rather than here.
seed_secret_from_env() {
  local secret="$1" env_key="$2"
  if gcloud secrets describe "${secret}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "    ${secret}: already exists, leaving it"
    return 0
  fi
  if [[ ! -f .env ]]; then
    echo "    ${secret}: no .env to read ${env_key} from — skipping" >&2
    return 1
  fi
  # python, not shell: these values contain & and other shell metacharacters.
  local tmp
  tmp="$(mktemp)"
  python3 -c "
import pathlib, re, sys
m = re.search(r'^${env_key}=(.*)\$', pathlib.Path('.env').read_text(), re.M)
v = (m.group(1).strip() if m else '')
sys.exit(1) if not v else pathlib.Path('${tmp}').write_text(v)
" || { echo "    ${secret}: ${env_key} is empty in .env — skipping"; rm -f "${tmp}"; return 1; }
  gcloud secrets create "${secret}" --replication-policy=automatic --project="${PROJECT_ID}" >/dev/null
  gcloud secrets versions add "${secret}" --data-file="${tmp}" --project="${PROJECT_ID}" >/dev/null
  rm -f "${tmp}"
  echo "    ${secret}: created from ${env_key}"
}

echo "==> Ensuring retrieval secrets exist"
seed_secret_from_env "mcp-database-url" "MCP_DATABASE_URL" || true
seed_secret_from_env "mcp-openai-api-key" "MCP_OPENAI_API_KEY" || true
seed_secret_from_env "mcp-tavily-api-key" "MCP_TAVILY_API_KEY" || true

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

echo "==> Granting ${RUNTIME_SA} access to the secrets"
for secret in "${SECRET_NAME}" mcp-database-url mcp-openai-api-key mcp-tavily-api-key; do
  gcloud secrets describe "${secret}" --project="${PROJECT_ID}" >/dev/null 2>&1 || continue
  gcloud secrets add-iam-policy-binding "${secret}" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role=roles/secretmanager.secretAccessor \
    --project="${PROJECT_ID}" >/dev/null
  echo "    ${secret}"
done

# Only mount secrets that actually exist, so a partial setup still deploys.
SECRET_MOUNTS="MCP_AUTH_TOKEN=${SECRET_NAME}:latest"
for pair in "MCP_DATABASE_URL=mcp-database-url" "MCP_OPENAI_API_KEY=mcp-openai-api-key" \
            "MCP_TAVILY_API_KEY=mcp-tavily-api-key"; do
  secret="${pair#*=}"
  if gcloud secrets describe "${secret}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    SECRET_MOUNTS="${SECRET_MOUNTS},${pair}:latest"
  fi
done

echo "==> Deploying"
# --allow-unauthenticated is deliberate: MCP clients can't mint Google ID
# tokens, so access is gated by the bearer token instead (see auth.py).
gcloud run deploy "${SERVICE}" \
  --source . \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --set-secrets="${SECRET_MOUNTS}" \
  --set-env-vars="MCP_SERVER_NAME=${SERVICE},MCP_LOG_LEVEL=info,MCP_OPENAI_BASE_URL=${EMBEDDING_BASE_URL},MCP_EMBEDDING_MODEL=${EMBEDDING_MODEL}" \
  --cpu=1 \
  --memory=512Mi \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=300

URL="$(gcloud run services describe "${SERVICE}" --region="${REGION}" \
  --project="${PROJECT_ID}" --format='value(status.url)')"

echo
echo "Deployed: ${URL}/mcp"
echo "Health:   curl -s ${URL}/health"
echo "Token:    gcloud secrets versions access latest --secret=${SECRET_NAME} --project=${PROJECT_ID}"

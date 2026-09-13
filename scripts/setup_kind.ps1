# Stands up a local kind cluster with ingress-nginx, kube-prometheus-stack, and
# Argo Rollouts, then applies the base k8s manifests. Prereqs: Docker Desktop, kind,
# kubectl, helm, and the kubectl-argo-rollouts plugin all installed and on PATH.
#
# Usage: ./scripts/setup_kind.ps1

$ErrorActionPreference = "Stop"

$ClusterName = "pdm"

Write-Host "Creating kind cluster '$ClusterName'..."
kind create cluster --name $ClusterName --config deploy/kind/kind-config.yaml

Write-Host "Installing ingress-nginx..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>$null
helm repo update ingress-nginx
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx `
  --namespace ingress-nginx --create-namespace `
  --set controller.hostPort.enabled=true `
  --set controller.service.type=ClusterIP `
  --set controller.nodeSelector."ingress-ready"="true"

Write-Host "Installing kube-prometheus-stack (Prometheus + Grafana + Alertmanager)..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>$null
helm repo update prometheus-community
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack `
  --namespace monitoring --create-namespace `
  --set grafana.sidecar.dashboards.enabled=true `
  --set grafana.sidecar.dashboards.label=grafana_dashboard

Write-Host "Installing Prometheus Pushgateway (for the drift-check CronJob)..."
helm upgrade --install prometheus-pushgateway prometheus-community/prometheus-pushgateway `
  --namespace monitoring

Write-Host "Installing Argo Rollouts controller..."
kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

Write-Host "Applying base pdm manifests (namespace, PVCs, MLflow, CronJobs, RBAC)..."
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc-mlflow.yaml
kubectl apply -f deploy/k8s/pvc-inference-log.yaml
kubectl apply -f deploy/k8s/mlflow-deployment.yaml
kubectl apply -f deploy/k8s/drift-check-rbac.yaml
kubectl apply -f deploy/k8s/retrain-cronjob.yaml
kubectl apply -f deploy/k8s/drift-check-cronjob.yaml

Write-Host "Applying Grafana dashboard/datasource provisioning ConfigMaps..."
kubectl create configmap pdm-grafana-dashboard `
  --namespace monitoring `
  --from-file=deploy/grafana/dashboards/pdm-overview.json `
  --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -

Write-Host "Applying Prometheus alert rules..."
kubectl apply -f monitoring/prometheus/alerts.yaml -n monitoring

Write-Host ""
Write-Host "Cluster ready. Next steps:"
Write-Host "  helm upgrade --install pdm-serving deploy/helm/pdm-serving -n pdm"
Write-Host "  kubectl argo rollouts get rollout pdm-serving -n pdm --watch"
Write-Host "  kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"

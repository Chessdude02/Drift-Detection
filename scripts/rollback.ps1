# Rolls back the pdm-serving Argo Rollout: aborts an in-progress canary (reverting to
# stable) or undoes to the previous stable revision if the rollout already finished.
#
# Usage:
#   ./scripts/rollback.ps1 abort      # cancel an in-progress canary, revert to stable now
#   ./scripts/rollback.ps1 undo       # roll back to the previous ReplicaSet revision

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("abort", "undo")]
    [string]$Action,

    [string]$Namespace = "pdm",
    [string]$Rollout = "pdm-serving"
)

$ErrorActionPreference = "Stop"

switch ($Action) {
    "abort" {
        Write-Host "Aborting in-progress canary for $Rollout in namespace $Namespace..."
        kubectl argo rollouts abort $Rollout -n $Namespace
    }
    "undo" {
        Write-Host "Rolling back $Rollout in namespace $Namespace to the previous revision..."
        kubectl argo rollouts undo $Rollout -n $Namespace
    }
}

kubectl argo rollouts get rollout $Rollout -n $Namespace

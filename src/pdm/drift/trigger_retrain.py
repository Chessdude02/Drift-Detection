"""Creates a one-off Kubernetes Job cloned from the retrain CronJob's pod template.

Runs only inside the cluster (uses in-cluster ServiceAccount credentials granted by
deploy/k8s/drift-check-rbac.yaml, scoped to create Jobs from that one CronJob's template).
"""

from __future__ import annotations

import logging
import time

from kubernetes import client, config

logger = logging.getLogger(__name__)


def trigger_retrain_job(namespace: str, source_cronjob: str, job_name_prefix: str) -> str:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()  # local dev / dry runs against a real kubeconfig

    batch_v1 = client.BatchV1Api()
    cronjob = batch_v1.read_namespaced_cron_job(name=source_cronjob, namespace=namespace)

    job_name = f"{job_name_prefix}-{int(time.time())}"
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={"triggered-by": "drift-check", "source-cronjob": source_cronjob},
        ),
        spec=cronjob.spec.job_template.spec,
    )
    created = batch_v1.create_namespaced_job(namespace=namespace, body=job)
    logger.info("Created retrain Job %s in namespace %s", created.metadata.name, namespace)
    return created.metadata.name

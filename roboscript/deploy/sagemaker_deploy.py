"""Session G — SageMaker deployment (PRD Section 6.9).

    python -m roboscript.deploy.sagemaker_deploy --dry-run       # print the plan, touch nothing
    python -m roboscript.deploy.sagemaker_deploy                 # create model/endpoint + alarms
    python -m roboscript.deploy.sagemaker_deploy --smoke-test    # invoke the endpoint with an ESTOP

Steps (each idempotent):
  1. image: {AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPOSITORY}:latest
     (build/push with the commands printed by --dry-run)
  2. SageMaker Model   (image + S3 model artifacts + IAM role)
  3. EndpointConfig    (ml.g4dn.xlarge by default)
  4. Endpoint          (waits for InService)
  5. CloudWatch alarms: 5xx error rate > 1 % and p95 ModelLatency > 500 ms -> SNS

Requires boto3 and AWS credentials; nothing here runs in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def config() -> dict:
    acct = os.getenv("AWS_ACCOUNT_ID", "000000000000")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    repo = os.getenv("ECR_REPOSITORY", "roboscript")
    return {
        "region": region,
        "image_uri": f"{acct}.dkr.ecr.{region}.amazonaws.com/{repo}:latest",
        "role_arn": os.getenv("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{acct}:role/SageMakerRole"),
        "instance_type": os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.g4dn.xlarge"),
        "model_data": f"s3://{os.getenv('S3_ARTIFACT_BUCKET', 'roboscript-artifacts')}/phi3-edge/model.tar.gz",
        "name": os.getenv("SAGEMAKER_ENDPOINT", "roboscript"),
        "sns_topic_arn": os.getenv("SNS_TOPIC_ARN", f"arn:aws:sns:{region}:{acct}:roboscript-alerts"),
        "env": {"ROBOSCRIPT_BACKEND": os.getenv("SM_BACKEND", "local"),
                "ROBOSCRIPT_EDGE_PATH": "/opt/ml/model",
                "ROBOSCRIPT_SFT_PATH": "/opt/ml/model",
                "ROBOSCRIPT_DPO_PATH": "/opt/ml/model"},
    }


def build_push_commands(c: dict) -> list[str]:
    reg = c["image_uri"].split("/")[0]
    return [f"aws ecr get-login-password --region {c['region']} | docker login --username AWS --password-stdin {reg}",
            "docker build -f docker/Dockerfile -t roboscript:latest .",
            f"docker tag roboscript:latest {c['image_uri']}",
            f"docker push {c['image_uri']}"]


def alarms(c: dict) -> list[dict]:
    dims = [{"Name": "EndpointName", "Value": c["name"]}, {"Name": "VariantName", "Value": "AllTraffic"}]
    return [
        {"AlarmName": f"{c['name']}-error-rate", "Namespace": "AWS/SageMaker",
         "Metrics": None, "MetricName": "Invocation5XXErrors", "Dimensions": dims,
         "Statistic": "Average", "Period": 300, "EvaluationPeriods": 1, "Threshold": 0.01,
         "ComparisonOperator": "GreaterThanThreshold", "AlarmActions": [c["sns_topic_arn"]],
         "TreatMissingData": "notBreaching"},
        {"AlarmName": f"{c['name']}-p95-latency", "Namespace": "AWS/SageMaker",
         "MetricName": "ModelLatency", "Dimensions": dims, "ExtendedStatistic": "p95",
         "Period": 300, "EvaluationPeriods": 1, "Threshold": 500_000,  # microseconds
         "ComparisonOperator": "GreaterThanThreshold", "AlarmActions": [c["sns_topic_arn"]],
         "TreatMissingData": "notBreaching"},
    ]


def deploy(c: dict) -> None:  # pragma: no cover - needs AWS
    import boto3
    sm = boto3.client("sagemaker", region_name=c["region"])
    cw = boto3.client("cloudwatch", region_name=c["region"])
    sm.create_model(ModelName=c["name"], ExecutionRoleArn=c["role_arn"],
                    PrimaryContainer={"Image": c["image_uri"], "ModelDataUrl": c["model_data"],
                                      "Environment": c["env"]})
    sm.create_endpoint_config(EndpointConfigName=c["name"], ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": c["name"], "InitialInstanceCount": 1,
        "InstanceType": c["instance_type"]}])
    sm.create_endpoint(EndpointName=c["name"], EndpointConfigName=c["name"])
    print("waiting for InService ...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=c["name"])
    for a in alarms(c):
        a = {k: v for k, v in a.items() if v is not None}
        cw.put_metric_alarm(**a)
    print(f"endpoint {c['name']} InService; alarms configured")


def smoke_test(c: dict) -> dict:  # pragma: no cover - needs AWS
    import boto3
    rt = boto3.client("sagemaker-runtime", region_name=c["region"])
    estop = {"yolo": None, "lidar": {"min_distance_meters": 2.0, "status": "CLEAR"},
             "telemetry": {"current_speed_mps": 1.0, "battery_level_pct": 65,
                           "active_fault_codes": ["ERR_ESTOP_TRIGGERED"]}}
    t0 = time.perf_counter()
    r = rt.invoke_endpoint(EndpointName=c["name"], ContentType="application/json",
                           Body=json.dumps(estop))
    out = json.loads(r["Body"].read())
    assert out["decision"] == "EMERGENCY_HALT" and out["ros2"]["cmd_vel"]["linear"]["x"] == 0.0
    print(f"smoke test OK in {(time.perf_counter() - t0) * 1000:.0f} ms")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()
    c = config()
    if a.dry_run:
        print("# build + push")
        print("\n".join(build_push_commands(c)))
        print("# sagemaker")
        print(json.dumps({k: v for k, v in c.items()}, indent=2))
        print("# alarms")
        print(json.dumps(alarms(c), indent=2))
        return
    if a.smoke_test:
        smoke_test(c)
        return
    deploy(c)


if __name__ == "__main__":
    main()

import mlflow


class MLflowTracker:
    def __init__(self, cfg):
        mlflow.set_tracking_uri(f"file://{cfg['mlflow']['tracking_dir']}")
        mlflow.set_experiment(cfg["mlflow"]["experiment"])

    def start_run(self, run_name, params):
        self._run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        return self._run

    def log_frame(self, metrics, frame_idx, frame_path=None):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=frame_idx)
        if frame_path:
            mlflow.log_artifact(frame_path, artifact_path="frames")

    def log_summary(self, summary_metrics):
        mlflow.log_metrics(summary_metrics)

    def end_run(self):
        run_id = mlflow.active_run().info.run_id
        mlflow.end_run()
        return run_id
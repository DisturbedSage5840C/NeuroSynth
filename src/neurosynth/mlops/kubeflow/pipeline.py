from __future__ import annotations

from kfp import dsl


@dsl.component(base_image="neurosynth:latest")
def ingest_patient_data(patient_id: str, data_sources: list[str]) -> str:
    _ = data_sources
    return f"s3://neurosynth/patients/{patient_id}/ingested.parquet"


@dsl.component(base_image="neurosynth:latest")
def preprocess_imaging(patient_data_path: str) -> str:
    _ = patient_data_path
    return "s3://neurosynth/tmp/connectome.pt"


@dsl.component(base_image="neurosynth:latest")
def run_gnn_inference(connectome_data_path: str) -> tuple[str, str]:
    _ = connectome_data_path
    return "s3://neurosynth/tmp/gnn_embedding.npy", "s3://neurosynth/tmp/gnn_preds.json"


@dsl.component(base_image="neurosynth:latest")
def run_genomic_inference(patient_data_path: str) -> str:
    _ = patient_data_path
    return "s3://neurosynth/tmp/genomic_embedding.npy"


@dsl.component(base_image="neurosynth:latest")
def run_tft_inference(patient_data_path: str) -> str:
    _ = patient_data_path
    return "s3://neurosynth/tmp/tft_forecast.json"


@dsl.component(base_image="neurosynth:latest")
def fuse_embeddings(gnn_emb: str, genomic_emb: str, tft_forecast: str) -> str:
    _ = (gnn_emb, genomic_emb, tft_forecast)
    return "s3://neurosynth/tmp/fused_repr.npy"


@dsl.component(base_image="neurosynth:latest")
def run_causal_discovery(patient_data_path: str, fused_repr: str) -> tuple[str, str]:
    _ = (patient_data_path, fused_repr)
    return "s3://neurosynth/tmp/causal_graph.json", "s3://neurosynth/tmp/interventions.json"


@dsl.component(base_image="neurosynth:latest")
def generate_clinical_report(fused_repr: str, causal_graph: str, interventions: str) -> str:
    _ = (fused_repr, causal_graph, interventions)
    return "s3://neurosynth/reports/report.json"


@dsl.pipeline(name="neurosynth_patient_analysis")
def neurosynth_patient_analysis(patient_id: str, data_sources: list[str]):
    ing = ingest_patient_data(patient_id=patient_id, data_sources=data_sources).set_retry(3)

    pre = preprocess_imaging(patient_data_path=ing.output).set_retry(3)
    pre.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)

    gnn = run_gnn_inference(connectome_data_path=pre.output).set_retry(3)
    gen = run_genomic_inference(patient_data_path=ing.output).set_retry(3)
    tft = run_tft_inference(patient_data_path=ing.output).set_retry(3)

    fuse = fuse_embeddings(gnn_emb=gnn.outputs["Output"], genomic_emb=gen.output, tft_forecast=tft.output).set_retry(3)
    causal = run_causal_discovery(patient_data_path=ing.output, fused_repr=fuse.output).set_retry(3)

    rep = generate_clinical_report(
        fused_repr=fuse.output,
        causal_graph=causal.outputs["Output"],
        interventions=causal.outputs["Output 1"],
    ).set_retry(3)

    for task in [ing, pre, gnn, gen, tft, fuse, causal, rep]:
        task.set_memory_limit("32Gi").set_cpu_limit("8")
        task.set_timeout("7200s")

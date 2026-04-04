from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NeuroSynthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEURO_", extra="ignore")

    log_level: str = Field(default="INFO")

    minio_endpoint: str = Field(default="http://minio:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_region: str = Field(default="us-east-1")

    iceberg_rest_uri: str = Field(default="http://iceberg-rest:8181")
    iceberg_warehouse: str = Field(default="s3://warehouse")

    timescale_dsn: str = Field(default="postgresql://postgres:postgres@timescaledb:5432/neurosynth")
    mimic_dsn: str = Field(default="postgresql://postgres:postgres@localhost:5432/mimic")

    neo4j_uri: str = Field(default="neo4j://neo4j:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    ppmi_base_url: str = Field(default="https://ida.loni.usc.edu/services/")
    ppmi_client_id: str = Field(default="")
    ppmi_client_secret: str = Field(default="")

    adni_sftp_host: str = Field(default="")
    adni_sftp_user: str = Field(default="")
    adni_sftp_password: str = Field(default="")

    mqtt_host: str = Field(default="localhost")
    mqtt_port: int = Field(default=1883)

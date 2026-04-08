from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_volume_3x3x3() -> np.ndarray:
    return np.arange(27, dtype=np.uint16).reshape(3, 3, 3)


@pytest.fixture
def synthetic_dicom_file(tmp_path: Path, synthetic_volume_3x3x3: np.ndarray) -> Path:
    pydicom_dataset = pytest.importorskip("pydicom.dataset")
    pydicom_uid = pytest.importorskip("pydicom.uid")

    Dataset = pydicom_dataset.Dataset
    FileDataset = pydicom_dataset.FileDataset
    ExplicitVRLittleEndian = pydicom_uid.ExplicitVRLittleEndian
    MRImageStorage = pydicom_uid.MRImageStorage
    generate_uid = pydicom_uid.generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_path = tmp_path / "synthetic_3x3x3.dcm"
    ds = FileDataset(str(dicom_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "MR"
    ds.Manufacturer = "NeuroSynthQA"
    ds.MagneticFieldStrength = 3.0
    ds.SeriesDescription = "Synthetic 3x3x3"
    ds.Rows = synthetic_volume_3x3x3.shape[1]
    ds.Columns = synthetic_volume_3x3x3.shape[2]
    ds.NumberOfFrames = synthetic_volume_3x3x3.shape[0]
    ds.ImagesInAcquisition = synthetic_volume_3x3x3.shape[0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PatientName = "Jane^Doe"
    ds.PatientID = "PHI-12345"
    ds.BurnedInAnnotation = "NO"
    ds.PixelData = synthetic_volume_3x3x3.tobytes()
    ds.save_as(dicom_path)
    return dicom_path


@pytest.fixture
def synthetic_vcf_text() -> str:
    return "\n".join(
        [
            "##fileformat=VCFv4.2",
            "##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
            "1\t1000\t.\tA\tG\t.\tPASS\tGENE=APOE",
            "1\t1100\t.\tC\tT\t.\tPASS\tGENE=TREM2",
        ]
    )


@pytest.fixture
def fake_iceberg() -> object:
    class _FakeIceberg:
        def __init__(self) -> None:
            self.records: dict[str, pd.DataFrame] = {}

        def append_dataframe(self, table_name: str, frame: pd.DataFrame) -> None:
            self.records[table_name] = frame.copy()

    return _FakeIceberg()


@pytest.fixture
def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    from backend import api as api_module

    async def _no_drain(timeout_seconds: int = 20) -> None:
        _ = timeout_seconds

    monkeypatch.setattr(api_module, "_drain_celery_queue", _no_drain)
    with TestClient(api_module.app) as client:
        yield client

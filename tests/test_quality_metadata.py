from pathlib import Path

from quality_intelligence.quality_metadata import infer_quality_metadata


def test_infer_quality_metadata_from_controlled_filename():
    metadata = infer_quality_metadata(
        Path("quality_knowledge_base/SOP__PlantaNorte__Empaque__SKU-100__SOP-QA-014__rev-03__2026-05-24.pdf")
    )

    assert metadata["document_type"] == "SOP"
    assert metadata["plant"] == "PlantaNorte"
    assert metadata["process"] == "Empaque"
    assert metadata["product"] == "SKU-100"
    assert metadata["document_code"] == "SOP-QA-014"
    assert metadata["revision"] == "03"
    assert metadata["document_date"] == "2026-05-24"


def test_infer_quality_metadata_classifies_customer_token():
    metadata = infer_quality_metadata(
        Path("AUDIT__PlantaSur__Liberacion__Cliente-ACME__AUD-2026-014__rev-00.pdf")
    )

    assert metadata["document_type"] == "AUDIT"
    assert metadata["customer"] == "Cliente-ACME"
    assert "product" not in metadata

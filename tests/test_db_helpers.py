from quality_intelligence.db import (
    build_filter_clause,
    chunk_metadata_column_values,
    document_metadata_json,
    metadata_column_values,
    validate_identifier,
    vector_literal,
)


def test_metadata_column_values_maps_aliases_and_dates():
    values = metadata_column_values(
        {
            "document_type": "sop",
            "code": "SOP-QA-014",
            "rev": "03",
            "plant": "PlantaNorte",
            "process": "Empaque",
            "document_date": "2026-05-24",
            "effective_date": "24/05/2026",
        }
    )

    assert values["document_type_code"] == "SOP"
    assert values["document_code"] == "SOP-QA-014"
    assert values["revision"] == "03"
    assert values["plant_code"] == "PlantaNorte"
    assert values["process_code"] == "Empaque"
    assert values["document_date"] == "2026-05-24"
    assert values["effective_date"] is None


def test_chunk_metadata_column_values_keeps_structured_fields():
    values = chunk_metadata_column_values(
        {
            "section_title": "Scope",
            "key_terms": ["CAPA", "SOP"],
            "detected_entities": {"capa": ["CAPA-2026-008"]},
        }
    )

    assert values["section_title"] == "Scope"
    assert values["key_terms"] == ["CAPA", "SOP"]
    assert values["detected_entities"] == {"capa": ["CAPA-2026-008"]}


def test_build_filter_clause_excludes_obsolete_by_default():
    where_sql, params = build_filter_clause(
        "quality_intelligence",
        {"plant": "PlantaNorte", "date_from": "2026-01-01"},
    )

    assert "COALESCE(d.is_current, TRUE) IS TRUE" in where_sql
    assert "ILIKE" in where_sql
    assert ">= %s::date" in where_sql
    assert params[0] == "quality_intelligence"
    assert "%PlantaNorte%" in params
    assert params[-1] == "2026-01-01"


def test_build_filter_clause_can_include_obsolete():
    where_sql, _ = build_filter_clause("quality_intelligence", {"include_obsolete": "true"})

    assert "COALESCE(d.is_current, TRUE) IS TRUE" not in where_sql


def test_document_metadata_json_is_sql_expression_not_function_call():
    expression = document_metadata_json("d")

    assert "jsonb_build_object" in expression
    assert "d.document_type_code" in expression
    assert "document_metadata_json" not in expression


def test_identifier_and_vector_literal_validation():
    assert validate_identifier("quality_intelligence") == "quality_intelligence"
    assert vector_literal([0.1, -0.2]) == "[0.1,-0.2]"

    try:
        validate_identifier("bad-name")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid SQL identifier to raise ValueError")

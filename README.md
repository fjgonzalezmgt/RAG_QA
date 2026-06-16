# Quality Intelligence Assistant

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-HNSW%2FIVFFLAT-2C5F2D?style=for-the-badge&logo=postgresql&logoColor=white)
![OpenAI compatible](https://img.shields.io/badge/OpenAI--compatible-API-412991?style=for-the-badge&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-env-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-PowerShell-0078D6?style=for-the-badge&logo=windows&logoColor=white)

Sistema RAG profesional para inteligencia documental y soporte operativo basado
en evidencia. El objetivo no es crear un chatbot generico, sino una herramienta
para consultar, recuperar, interpretar y conectar informacion tecnica de calidad,
operaciones, Lean Six Sigma, manufactura, supply chain y QMS.

La base operativa del sistema es PostgreSQL (`RAG_DB`) y el esquema principal
del producto es `quality_intelligence`.

## Para que sirve

El asistente esta orientado a documentos como:

- SOPs y procedimientos.
- CAPA, desviaciones y no conformidades.
- Auditorias internas, externas, de cliente o proveedor.
- Reclamos de cliente.
- Especificaciones de producto, proceso, empaque o cliente.
- Reportes de calidad.
- Lecciones aprendidas.
- Proyectos DMAIC.
- Indicadores operativos.
- Documentacion QMS.

El sistema responde con evidencia recuperada, citas por documento/pagina,
contexto operativo y una estructura util para tomar decisiones.

## Business case

### Problema

En operaciones reales de calidad y manufactura, la informacion critica suele
estar distribuida entre procedimientos, especificaciones, CAPA, auditorias,
reclamos, reportes, indicadores y documentos QMS. Esto provoca:

- busquedas manuales lentas antes de tomar decisiones;
- dificultad para conectar reclamos, causas raiz, CAPA y auditorias;
- uso de documentos obsoletos o no aplicables;
- perdida de conocimiento entre plantas, turnos o equipos;
- preparacion reactiva ante auditorias;
- decisiones operativas con evidencia incompleta;
- alto tiempo invertido por ingenieros, coordinadores y gerentes en rastrear
  informacion documental.

### Propuesta de valor

El **Quality Intelligence Assistant** centraliza la inteligencia documental del
sistema de calidad y permite consultar evidencia tecnica con contexto operativo:
planta, proceso, producto, cliente, auditoria, tipo documental y fecha.

No reemplaza al responsable de calidad ni aprueba decisiones de forma automatica.
Su valor es acelerar el analisis, mostrar evidencia trazable y reducir el riesgo
de decisiones sin soporte documental.

### Usuarios objetivo

- Gerentes de calidad y operaciones.
- Ingenieros de calidad, proceso y mejora continua.
- Lideres Lean Six Sigma.
- Auditores internos.
- Responsables CAPA.
- Equipos de manufactura, supply chain y servicio al cliente.
- Analistas de QMS y documentacion.

### Casos de uso empresariales

- Preparacion de auditorias con evidencia por proceso, planta y requisito.
- Investigacion de reclamos conectando especificaciones, historial y CAPA.
- Analisis de recurrencia de no conformidades o hallazgos.
- Consulta rapida de SOPs vigentes y registros requeridos.
- Revision de riesgos antes de cambios de proceso o producto.
- Transferencia de lecciones aprendidas entre plantas.
- Soporte documental para proyectos DMAIC.
- Analisis de indicadores operativos con contexto de eventos de calidad.

### Impacto esperado

- Menor tiempo de busqueda documental.
- Mejor trazabilidad entre decision, evidencia y fuente.
- Mayor reutilizacion de conocimiento tecnico.
- Preparacion mas ordenada para auditorias.
- Reduccion de respuestas inconsistentes entre areas o plantas.
- Identificacion mas rapida de brechas documentales.
- Mejor soporte para priorizar CAPA, acciones preventivas y proyectos de mejora.

### Indicadores para medir valor

- Tiempo promedio para encontrar evidencia documental.
- Porcentaje de respuestas con cita valida por pagina/documento.
- Reduccion de hallazgos por falta de evidencia o documento incorrecto.
- Tiempo de preparacion de auditoria.
- Reutilizacion de CAPA/lecciones aprendidas en nuevos eventos.
- Porcentaje de consultas filtradas por planta, proceso, producto o cliente.
- Numero de brechas documentales detectadas antes de auditorias.

## Preguntas tipo

Estas preguntas muestran el tipo de interaccion esperada. El objetivo es obtener
respuestas basadas en evidencia, no opiniones genericas.

### SOPs y procedimientos

- Que procedimiento aplica para liberar este producto en esta planta?
- Que evidencias exige el SOP para cerrar esta etapa?
- Que registros son obligatorios segun el procedimiento?
- Quien es responsable de aprobar o ejecutar esta actividad?
- Que pasos del procedimiento impactan directamente al cliente?
- Hay diferencias entre el SOP actual y una revision anterior?
- Que controles se deben verificar antes de iniciar produccion?

### CAPA, causa raiz y recurrencia

- Que CAPA anteriores tuvieron una causa raiz parecida?
- Que acciones correctivas se usaron antes para este mismo defecto?
- Que evidencia existe de que la accion fue efectiva?
- Esta causa raiz relacionada con algun hallazgo de auditoria?
- Hay recurrencia por proceso, producto, cliente o planta?
- Que brechas de informacion impiden cerrar esta CAPA?

### Auditorias y cumplimiento

- Que hallazgos de auditoria se repiten por proceso?
- Que documentos debo revisar antes de una auditoria externa?
- Que evidencia necesito preparar para este requisito?
- Que hallazgos abiertos afectan a esta planta?
- Que procedimientos soportan este punto de la auditoria?
- Existe evidencia documental suficiente para demostrar cumplimiento?

### Reclamos de cliente

- Que especificacion del cliente define el criterio de aceptacion?
- Que reclamos similares se han recibido para este producto?
- Que CAPA o acciones se relacionan con este reclamo?
- Que lote, producto, proceso o planta aparece relacionado en la evidencia?
- Que respuesta tecnica puede sustentarse con documentos existentes?
- Que informacion falta antes de responder al cliente?

### Especificaciones y liberacion

- Cual es el criterio de aceptacion aplicable?
- Que especificacion vigente aplica a este producto y cliente?
- Que tolerancias o limites deben verificarse?
- Que documentos soportan la decision de aceptar o rechazar un lote?
- Hay conflicto entre especificacion interna y requerimiento de cliente?
- Que controles del proceso estan asociados a esta especificacion?

### Riesgo y decisiones operativas

- Que riesgos aparecen si cambio este parametro de proceso?
- Que impacto potencial existe para cliente, producto, proceso o compliance?
- Que evidencia soporta una decision de contencion?
- Que decision recomendada se puede tomar con la evidencia disponible?
- Que documentos estan vencidos, incompletos o sin aprobacion?
- Que brechas de informacion impiden responder con confianza?

### DMAIC, mejora continua e indicadores

- Que proyectos DMAIC han tratado un problema similar?
- Que causas potenciales se identificaron en proyectos anteriores?
- Que controles quedaron como parte del plan de control?
- Que indicadores muestran deterioro en este proceso?
- Que eventos de calidad coinciden con cambios en el KPI?
- Que lecciones aprendidas aplican a este nuevo proyecto?

## Funcionalidades principales

- Ingesta de PDFs con extraccion por pagina, hash SHA-256 y deduplicacion por
  contenido.
- Chunking con solape configurable y conservacion de `page_start` / `page_end`.
- Embeddings OpenAI o locales almacenados en PostgreSQL + pgvector.
- Selector de proveedor IA en Streamlit para alternar entre OpenAI y LM Studio
  local compatible con la API de OpenAI.
- Perfil especializado `quality_intelligence` para respuestas en contexto de
  calidad, manufactura, mejora continua y QMS. La UI esta fijada a este dominio
  para evitar mezclar esquemas o perfiles no validados.
- Filtros operativos en Streamlit:
  - planta / sitio
  - proceso / area
  - producto / SKU
  - cliente
  - tipo documental
  - auditoria / hallazgo
  - rango de fechas
- Opciones de filtros operativos generadas automaticamente desde la metadata
  indexada, con entrada manual como fallback.
- Retrieval semantico con diversificacion por documento.
- Citas trazables tipo `[S1] Documento, pp. 4-5`, con metadata de revision,
  vigencia, aprobacion y contexto operativo cuando esta disponible.
- Panel de evidencia con score, revision, vigencia, aprobacion, tipo documental,
  paginas, contexto operativo y extracto.
- Semaforo de confiabilidad documental basado en cantidad de evidencia,
  vigencia, aprobacion, revision y fechas.
- Modos de consulta: procedimiento, CAPA/causa raiz, auditoria, reclamo,
  especificacion e indicadores/KPI.
- Preguntas rapidas para resumen documental, brechas, auditoria, comparacion de
  versiones y CAPA similares.
- Explorador documental, vista de brechas y historial trazable de sesiones.
- Exportacion de briefing en Markdown con respuesta, brechas y fuentes.
- Inferencia inicial de metadata QMS desde nombres controlados de archivo como
  fallback.
- Enriquecimiento opcional de metadata desde el contenido del PDF usando LLM.
- Esquema SQL extendido para trazabilidad, evidencia, eventos de calidad,
  auditorias, DMAIC, indicadores y decisiones.
- Script para inicializar el esquema profesional `quality_intelligence`.

## Arquitectura

```mermaid
flowchart LR
    U["Usuario de calidad / operaciones"] --> ST["Streamlit app.py"]
    U --> CLI["CLI scripts"]

    ST --> ING["PDFIngestor"]
    ST --> RET["RAGRetriever"]
    ST --> LLM["LLMClient"]
    CLI --> ING

    ING --> PDF["PDF loader"]
    ING --> SPL["Text splitter"]
    ING --> META["Quality metadata<br/>reglas + LLM opcional"]
    ING --> EMB["EmbeddingClient"]
    ING --> PG[("PostgreSQL RAG_DB<br/>quality_intelligence + pgvector")]

    RET --> EMB
    RET --> PG
    LLM --> OAI["Proveedor IA<br/>OpenAI o LM Studio"]
    EMB --> OAI

    PG --> EV["Evidencia trazable<br/>documento + pagina + metadata"]
    EV --> LLM
```

Flujo de trabajo:

1. El usuario coloca documentos tecnicos en `quality_knowledge_base/`.
2. La ingesta extrae texto, calcula hash, infiere metadata, opcionalmente la
   enriquece con LLM desde el contenido del PDF y genera chunks.
3. Los chunks se convierten en embeddings y se guardan en pgvector.
4. El usuario consulta usando filtros operativos.
5. El retriever combina similitud semantica, filtros y diversidad documental.
6. El LLM genera una respuesta basada solo en evidencia recuperada.
7. La UI muestra respuesta, semaforo documental, citas, fuentes, brechas,
   historial trazable y briefing exportable.

## Estructura del proyecto

- `app.py`: interfaz Streamlit del Quality Intelligence Assistant.
- `quality_knowledge_base/`: carpeta recomendada para documentos tecnicos a ingerir.
- `scripts/ingest_pdfs.py`: ingesta batch de PDFs.
- `scripts/init_quality_schema.py`: inicializa el esquema profesional QMS.
- `sql/001_quality_intelligence_schema.sql`: migracion PostgreSQL extendida.
- `docs/quality_intelligence_architecture.md`: arquitectura funcional completa,
  metadata, retrieval, roadmap, riesgos y buenas practicas.
- `src/quality_intelligence/config.py`: configuracion desde `.env`.
- `src/quality_intelligence/db.py`: persistencia PostgreSQL, pgvector, indices, filtros y
  busqueda semantica.
- `src/quality_intelligence/ingest.py`: pipeline de ingesta.
- `src/quality_intelligence/quality_metadata.py`: inferencia de metadata QMS desde nombres
  de archivo.
- `src/quality_intelligence/metadata_enrichment.py`: extraccion opcional de
  metadata desde contenido documental usando LLM y JSON validado.
- `src/quality_intelligence/retriever.py`: retrieval y etiquetado de fuentes.
- `src/quality_intelligence/llm.py`: prompt final y llamada al modelo.
- `src/quality_intelligence/domain_profiles.py`: perfiles de dominio, incluyendo
  `quality_intelligence`.
- `src/quality_intelligence/pdf_loader.py`: lectura y hashing de PDFs.
- `src/quality_intelligence/text_splitter.py`: chunking con rango de paginas.
- `src/quality_intelligence/embeddings.py`: cliente de embeddings OpenAI-compatible.

## Modelo de datos

El modelo de datos esta disenado alrededor de dos tablas documentales centrales:

- `quality_intelligence.documents`
- `quality_intelligence.chunks`

Sobre esa base documental, el esquema incorpora tablas para contexto operativo,
trazabilidad, eventos de calidad, evidencia y decisiones:

- `document_types`: clasificacion documental.
- `plants`, `processes`, `products`, `customers`: dimensiones operativas.
- `quality_events`: CAPA, no conformidades, desviaciones, reclamos y hallazgos.
- `document_event_links`: relacion entre documentos y eventos.
- `audit_records`: auditorias y hallazgos.
- `dmaic_projects`: proyectos Lean Six Sigma.
- `operational_indicator_measurements`: KPIs por periodo.
- `retrieval_sessions`: preguntas, filtros y respuestas.
- `retrieval_evidence`: chunks usados como evidencia.
- `decision_records`: decisiones, racional, riesgo y evidencia asociada.

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : contains
    DOCUMENTS ||--o{ DOCUMENT_EVENT_LINKS : supports
    QUALITY_EVENTS ||--o{ DOCUMENT_EVENT_LINKS : relates
    RETRIEVAL_SESSIONS ||--o{ RETRIEVAL_EVIDENCE : cites
    CHUNKS ||--o{ RETRIEVAL_EVIDENCE : evidence
    RETRIEVAL_SESSIONS ||--o{ DECISION_RECORDS : informs

    DOCUMENTS {
        uuid id PK
        text domain
        text file_name
        text content_hash
        text document_type_code
        text revision
        date effective_date
        text plant_code
        text process_code
        text product_code
        text customer_code
        jsonb metadata
    }
    CHUNKS {
        uuid id PK
        uuid document_id FK
        int chunk_index
        int page_start
        int page_end
        text content
        vector embedding
        jsonb metadata
    }
    QUALITY_EVENTS {
        text event_code PK
        text event_type
        text severity
        text root_cause
        text corrective_action
        text status
    }
```

## Metadata recomendada

Metadata minima por documento:

- `document_type`: SOP, CAPA, AUDIT, COMPLAINT, SPECIFICATION, DMAIC, KPI, QMS.
- `document_code`, `revision`, `lifecycle_status`.
- `document_date`, `effective_date`, `review_due_date`.
- `plant`, `process`, `product`, `customer`.
- `owner_area`, `approval_status`, `approved_by`.
- `qms_process`, `risk_level`, `source_system`, `source_record_id`.

Metadata por chunk:

- `page_start`, `page_end`.
- `section_title`, `section_number`, `clause_ref`.
- `requirement_type`.
- `process_step`, `risk_signal`, `key_terms`.
- IDs detectados: CAPA, auditoria, reclamo, lote, SKU o cliente.

## Metadata automatica

La metadata puede venir de tres fuentes, en este orden practico:

1. **Contenido del PDF con LLM**, cuando `RAG_LLM_METADATA_ENRICHMENT=true` o se
   activa desde la UI. El modelo recibe una muestra del texto del documento y
   devuelve JSON validado. El nombre del archivo se usa solo para trazabilidad,
   no como evidencia.
2. **Reglas deterministicas**, como fechas, codigos, secciones, entidades CAPA,
   auditorias, SKU o lotes detectados en texto.
3. **Nombre del archivo**, como fallback para demos o carpetas sin maestro
   documental.

Campos soportados a nivel documento:

- `document_type`, `document_code`, `revision`, `lifecycle_status`.
- `document_date`, `effective_date`, `review_due_date`.
- `plant`, `process`, `product`, `customer`.
- `owner_area`, `approval_status`, `approved_by`, `approved_at`.
- `qms_process`, `risk_level`, `source_system`, `source_record_id`.
- `confidentiality_level`.

La metadata sugerida por LLM se guarda junto con una traza `llm_metadata` que
incluye confianza, campos aceptados, campos sobrescritos y notas de evidencia.

### Convencion de nombres para demo

Si no usas enriquecimiento LLM ni metadata externa, puedes colocar PDFs en
`quality_knowledge_base/` con esta convencion:

```text
<document_type>__<plant>__<process>__<product-or-customer>__<code>__rev-<revision>.pdf
```

Ejemplo:

```text
SOP__PlantaNorte__Empaque__SKU-100__SOP-QA-014__rev-03.pdf
CAPA__PlantaNorte__Empaque__SKU-100__CAPA-2026-008__rev-01.pdf
AUDIT__PlantaSur__Liberacion__Cliente-ACME__AUD-2026-014__rev-00.pdf
```

En produccion, lo ideal es combinar LLM con metadata maestra de QMS, SharePoint,
ERP, MES, LIMS, CRM, sistema de auditorias o sistema CAPA.

## Configuracion

`.env` contiene la configuracion local y no debe versionarse. `.env.sample`
contiene una plantilla segura.

Valores principales:

```text
DB_NAME=RAG_DB
RAG_DOMAIN=quality_intelligence
RAG_PDF_DIR=./quality_knowledge_base
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIM=768
AI_PROVIDER=openai
```

Para usar modelos locales con LM Studio, inicia el servidor local compatible con
OpenAI, carga los modelos y usa:

```text
AI_PROVIDER=lm_studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_CHAT_MODEL=nvidia/nemotron-3-nano-4b
LM_STUDIO_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v2-moe
LM_STUDIO_EMBEDDING_DIM=768
```

La configuracion recomendada mantiene compatibilidad entre OpenAI, modelos
locales servidos por LM Studio y PostgreSQL + pgvector usando la misma dimension
de embeddings en ambos proveedores. En este README se muestra una configuracion
de referencia con 768 dimensiones para alternar entre OpenAI y LLM local sin
modificar el indice vectorial.

Variables de base de datos:

- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.
- `DB_SSLMODE`.
- `DB_EXTENSIONS_SCHEMA`, por defecto `extensions`.

Variables RAG:

- `RAG_DOMAIN`: esquema/dominio PostgreSQL. La UI fuerza
  `quality_intelligence`; esta variable queda para scripts o compatibilidad.
- `RAG_PDF_DIR`: carpeta raiz con PDFs.
- `RAG_CHUNK_SIZE` y `RAG_CHUNK_OVERLAP`.
- `RAG_TOP_K`, `RAG_CANDIDATE_K`, `RAG_MAX_CHUNKS_PER_DOCUMENT`.
- `RAG_MAX_CONTEXT_CHARS`.
- `RAG_RECURSIVE_PDF_SCAN`: incluye subcarpetas dentro del repositorio documental.
- `RAG_PDF_TEXT_FALLBACK`: intenta fallback con `pdftotext` y, si esta instalado,
  OCR opcional con `ocrmypdf` cuando `pypdf` no extrae texto.
- `RAG_LLM_METADATA_ENRICHMENT`: activa extraccion de metadata desde el contenido
  del PDF usando LLM durante la ingesta.
- `RAG_LLM_METADATA_MAX_CHARS`: maximo de texto del documento enviado al LLM para
  extraer metadata documental.

Variables del proveedor IA:

- `AI_PROVIDER`: `openai` o `lm_studio`.
- `OPENAI_*`: endpoint, clave, modelos y parametros para OpenAI.
- `LM_STUDIO_*`: endpoint, clave local ficticia, modelos y prefijos de embeddings
  para LM Studio.
- Los prefijos de embeddings son opcionales: si no los defines, el sistema usa
  defaults para Nomic v2 MoE.
- Desde la UI puedes cambiar proveedor, endpoint, modelo de chat, modelo de
  embeddings, dimension y prefijos sin editar `.env`.

## Instalacion

Crear y activar el entorno:

```powershell
conda env create -f environment.yml
conda activate quality-intelligence
```

El archivo `environment.yml` crea el entorno `quality-intelligence`. Si prefieres
usar el entorno local existente `rag_qa`, instala las dependencias ahi:

En este workspace tambien se ha validado un entorno Conda llamado `rag_qa`:

```powershell
conda install -n rag_qa -c conda-forge openai loguru psycopg python-dotenv pypdf streamlit poppler pytest -y
```

`poppler` instala `pdftotext`, usado como fallback de extraccion de texto.
`ocrmypdf` es opcional; en Windows puede requerir dependencias adicionales.
Tambien puedes usar `requirements.txt` con `pip` si prefieres otro entorno.

## Preparar PostgreSQL

La base objetivo es `RAG_DB`. El usuario configurado debe poder conectarse a
esa base y crear objetos dentro de ella.

Si la base no existe, creala con un usuario administrador:

```sql
CREATE DATABASE "RAG_DB";
```

Luego aplica el esquema profesional:

```powershell
conda run -n rag_qa python scripts\init_quality_schema.py --db-name RAG_DB
```

Si el usuario tiene permiso para crear bases:

```powershell
conda run -n rag_qa python scripts\init_quality_schema.py --db-name RAG_DB --create-db
```

Desde Streamlit puedes usar **Inicializar BD** para preparar el indice
documental. Para una instalacion completa del producto, ejecuta la migracion SQL
porque crea tambien las dimensiones operativas, eventos, auditorias, evidencia,
indicadores y decisiones.

## Ingestar documentos

Coloca PDFs en `quality_knowledge_base/` y ejecuta:

```powershell
conda run -n rag_qa python scripts\ingest_pdfs.py --pdf-dir .\quality_knowledge_base --domain quality_intelligence
```

Para reemplazar chunks de archivos ya indexados:

```powershell
conda run -n rag_qa python scripts\ingest_pdfs.py --pdf-dir .\quality_knowledge_base --domain quality_intelligence --force
```

Tambien puedes usar el boton **Ingerir PDFs** desde Streamlit.

## Ejecutar Streamlit

```powershell
conda run -n rag_qa streamlit run app.py --server.port 8501
```

O ejecuta `run_app.bat` en Windows.

La interfaz incluye:

- configuracion de carpeta documental;
- controles de chunking, retrieval y enriquecimiento de metadata con LLM;
- prueba de conexion con el proveedor IA activo;
- filtros operativos y toggle para incluir documentos obsoletos;
- filtros dinamicos basados en metadata indexada;
- modos de consulta por caso operativo;
- preguntas rapidas;
- explorador de documentos indexados;
- vista de brechas de metadata;
- historial trazable de sesiones;
- chat operacional;
- semaforo de confiabilidad documental;
- fuentes con score, documento, pagina, revision, vigencia, aprobacion,
  metadata y extracto;
- exportacion de briefing en Markdown.

## Tests

La suite cubre helpers de metadata, chunking, filtros SQL, prompt/contexto LLM y
enriquecimiento de metadata sin hacer llamadas reales al proveedor IA.

Ejecutar:

```powershell
conda run -n rag_qa pytest
```

Verificacion esperada:

```text
18 passed
```

## Buenas practicas de uso

- Usa filtros operativos antes de consultar temas de alto riesgo.
- Revisa siempre las fuentes citadas, especialmente revision y vigencia.
- No trates la respuesta como aprobacion formal automatica.
- Declara brechas cuando falten documentos, paginas, fechas o estado de
  aprobacion.
- Mantiene documentos obsoletos separados o marcados con metadata.
- Usa preguntas doradas para evaluar periodicamente la calidad del retrieval.

## Limitaciones y riesgos

- PDFs escaneados requieren OCR antes de la ingesta.
- Metadata incompleta reduce la precision de filtros.
- La similitud semantica puede recuperar documentos parecidos pero no aplicables.
- Documentos obsoletos pueden contaminar respuestas si no se controla vigencia.
- Causa raiz, riesgo y decision final requieren validacion humana.
- Informacion sensible de cliente, producto o proceso requiere controles de
  acceso antes de un despliegue empresarial.

## Roadmap sugerido

1. Busqueda hibrida: combinar vector search con busqueda textual para codigos,
   lotes, SKUs, normas y clausulas.
2. Reranking especializado para priorizar evidencia vigente, aprobada y
   aplicable al contexto operativo.
3. Revision humana de metadata: pantalla para aceptar, corregir o rechazar
   metadata sugerida por LLM antes de marcarla como aprobada.
4. Registro avanzado de decisiones: responsables, fechas, riesgo, estado y
   evidencia asociada.
5. Demo empresarial: dataset simulado realista con casos de auditoria, reclamo,
   CAPA y DMAIC.
6. Producto Quality Analytics: conectores QMS/SharePoint/ERP, permisos, evals,
   monitoreo y despliegue controlado.

## Documentacion ampliada

La arquitectura funcional completa esta en:

```text
docs/quality_intelligence_architecture.md
```

Incluye detalle sobre tablas PostgreSQL, metadata QMS, tipos documentales,
flujos de retrieval, preguntas esperadas, trazabilidad, riesgo, decisiones,
roadmap, riesgos tecnicos y buenas practicas de embeddings/chunking.

## Licencia

Este repositorio es publico solo como demo, portafolio y material de evaluacion.
El codigo es **source-available**, pero no es open source.

No se concede permiso para usar, copiar, modificar, distribuir, desplegar,
revender, alojar, convertir en SaaS, incorporar en otro producto ni usar este
software en entornos comerciales, internos, de consultoria, capacitacion u
operacion sin autorizacion escrita previa.

El uso comercial, modificacion, redistribucion, despliegue, uso interno de
negocio, consultoria, capacitacion o incorporacion en otro producto requiere una
licencia comercial pagada.

Consulta `LICENSE.md` para los terminos completos.

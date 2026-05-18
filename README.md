# RAG Books

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-HNSW%2FIVFFLAT-2C5F2D?style=for-the-badge&logo=postgresql&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-env-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![Loguru](https://img.shields.io/badge/Loguru-logging-000000?style=for-the-badge&logo=python&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-PowerShell-0078D6?style=for-the-badge&logo=windows&logoColor=white)

Sistema **RAG (Retrieval-Augmented Generation)** parametrizable para libros literarios o documentos de otros dominios (por ejemplo, sistemas de gestion).
La base de datos es unica (`RAG_DB`) y cada dominio vive en su propio esquema PostgreSQL, lo que permite indexar y consultar corpus separados sin mezclar contextos.

## Arquitectura

```mermaid
flowchart LR
    subgraph User["Usuario"]
        U[Persona]
    end

    subgraph UI["Interfaces"]
        ST["Streamlit app.py"]
        CLI["CLI scripts/ingest_pdfs.py"]
    end

    subgraph Core["Nucleo rag_books"]
        ING["PDFIngestor<br/>ingest.py"]
        PDF["PDF loader<br/>pdf_loader.py"]
        SPL["Text splitter<br/>text_splitter.py"]
        EMB["EmbeddingClient<br/>embeddings.py"]
        RET["RAGRetriever<br/>retriever.py"]
        LLM["LLMClient<br/>llm.py"]
        PROF["Domain profiles<br/>domain_profiles.py"]
        CFG["Settings<br/>config.py"]
    end

    subgraph External["Servicios externos"]
        OAI["OpenAI API<br/>chat + embeddings"]
        PG[("PostgreSQL<br/>pgvector")]
    end

    U --> ST
    U --> CLI
    ST --> ING
    ST --> RET
    ST --> LLM
    CLI --> ING

    ING --> PDF
    ING --> SPL
    ING --> EMB
    ING --> PG

    RET --> EMB
    RET --> PG
    LLM --> PROF
    LLM --> OAI
    EMB --> OAI

    CFG -.->|inyecta config| ING
    CFG -.->|inyecta config| RET
    CFG -.->|inyecta config| LLM
```

## Stack

- **PostgreSQL** con extension `vector` (pgvector) para busqueda semantica con indices HNSW (fallback automatico a IVFFLAT).
- **Python 3.11+** como lenguaje principal.
- **OpenAI API** para embeddings (`text-embedding-3-large`, 2000 dim por defecto) y respuestas del LLM (`gpt-5.5` por defecto).
- **Streamlit** como interfaz web con chat, configuracion en barra lateral, panel de indice y vista de fuentes.
- **pypdf** para extraccion de texto pagina por pagina.
- **psycopg 3** como driver PostgreSQL.
- **Conda** para gestion del entorno (`environment.yml`).
- **Loguru** para logging estructurado con rotacion en disco.

## Licencia

Este proyecto se distribuye bajo la licencia Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
Consulta el archivo `LICENSE.md` para los términos completos.

## Funcionalidad

### Ingesta de PDFs

- Descubrimiento de PDFs en la **raiz** de la carpeta configurada (no recursivo).
- Extraccion de texto por pagina con `pypdf`, incluyendo metadatos de titulo y autor cuando estan presentes.
- **Deduplicacion por contenido**: cada PDF se identifica con un hash SHA-256; si el archivo no cambio, se omite la reingesta.
- **Modo `--force`** para reemplazar todos los chunks de un archivo y reingestar desde cero.
- Particion en chunks con tamano y solape configurables (`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`), con ajuste a fronteras naturales (oracion, parrafo, salto de linea).
- Conservacion del rango de paginas por chunk para citas precisas.

```mermaid
flowchart TD
    A[list_pdfs<br/>carpeta raiz] --> B{Para cada PDF}
    B --> C[load_pdf<br/>texto por pagina + SHA-256]
    C --> D{force?}
    D -->|si| E[delete_documents_by_source]
    D -->|no| F{Hash ya<br/>indexado?}
    F -->|si| G[Skip<br/>documents_skipped++]
    F -->|no| H[split_pages<br/>chunks con solape]
    E --> H
    H --> I{Hay texto?}
    I -->|no| J[Registrar error]
    I -->|si| K[embed_texts<br/>batches OpenAI]
    K --> L[insert_document<br/>titulo + autor + hash]
    L --> M[insert_chunks<br/>contenido + embedding + paginas]
    M --> N[documents_ingested++<br/>chunks_created += N]
    G --> B
    J --> B
    N --> B
    B -->|fin| O[IngestResult<br/>resumen + errores]
```

### Embeddings y almacenamiento

- Llamadas en lotes a la API de embeddings con limite por cantidad de textos y por caracteres totales.
- Soporte para dimensiones personalizadas en modelos `text-embedding-3-*`.
- Tabla `documents` y tabla `chunks` por esquema/dominio, con `UNIQUE` por `(domain, source_path, content_hash)` y por `(document_id, chunk_index)`.
- **Validacion automatica de dimension**: si la dimension de embedding cambia y las tablas estan vacias, se recrean; si tienen datos, se aborta con instrucciones claras.
- Indice **HNSW** sobre `embedding` con metrica coseno; fallback a **IVFFLAT** si la version de pgvector no soporta HNSW; si ambos fallan, se usa busqueda exacta.

### Recuperacion (retriever)

- Embedding de la pregunta del usuario y busqueda kNN por distancia coseno (`<=>`).
- **Diversificacion por documento**: se buscan `RAG_CANDIDATE_K` candidatos, se limitan a `RAG_MAX_CHUNKS_PER_DOCUMENT` por documento y se rellena hasta `RAG_TOP_K` con los siguientes mejores.
- Cada chunk recuperado se etiqueta con un id de cita (`S1`, `S2`, ...) que incluye titulo o nombre de archivo y rango de paginas.

```mermaid
sequenceDiagram
    actor U as Usuario
    participant ST as Streamlit
    participant R as RAGRetriever
    participant E as EmbeddingClient
    participant DB as pgvector
    participant L as LLMClient
    participant O as OpenAI

    U->>ST: Escribe pregunta
    ST->>R: retrieve(question, top_k, candidate_k)
    R->>E: embed_query(question)
    E->>O: POST /embeddings
    O-->>E: vector
    E-->>R: query_embedding

    alt candidate_k > top_k
        R->>DB: search_diverse(top_k, candidate_k, max_per_doc)
        DB-->>R: candidatos kNN
        R->>R: Limitar por documento<br/>y rellenar hasta top_k
    else
        R->>DB: search(top_k)
        DB-->>R: kNN exacto
    end

    R-->>ST: contextos [S1..Sk] con citas
    ST->>L: answer(question, contextos, profile, history)
    L->>L: build_context_block<br/>build_history_block
    L->>O: POST /chat/completions
    O-->>L: respuesta
    L-->>ST: texto con citas [S1], [S2]
    ST-->>U: respuesta + expansor Fuentes
```

### Generacion de respuesta (LLM)

- Prompts especializados por **dominio** (perfiles `literatura` y `sistemas_gestion` integrados, mas un perfil **personalizado** editable desde la UI).
- Construccion del prompt con: instruccion de sistema/desarrollador, historial reciente de la conversacion, bloque de contexto recuperado con citas y la pregunta del usuario.
- **Memoria conversacional corta** en Streamlit: el historial se envia al LLM solo para resolver referencias, no como evidencia documental.
- Parametros compatibles segun modelo: `reasoning_effort` y `verbosity` para la familia GPT-5; `temperature` solo para modelos que la soportan.
- Rol de instruccion `developer` para `gpt-5*`/`o*`, `system` para el resto.

### Interfaz Streamlit (`app.py`)

Barra lateral:
- Carpeta de PDFs.
- Selector de dominio (literatura, sistemas de gestion o personalizado con prompt editable).
- Sliders: chunks recuperados (`top_k`), candidatos para diversificar, max chunks por documento.
- Tamano y solape de chunk para reingestas.
- Estado de conexion a OpenAI y boton **Probar OpenAI API**.
- Boton **Limpiar conversacion**.

Panel izquierdo (Indice):
- **Inicializar BD**: crea esquema, tablas e indices.
- **Ingerir PDFs**: ejecuta la ingesta con mensajes de progreso en vivo.
- Tabla con documentos indexados y conteo de chunks.

Panel derecho (Consulta):
- Chat con historial visible.
- Cada respuesta del asistente incluye un expansor **Fuentes** con cita, score, archivo y extracto del chunk.

### CLI

Script `scripts/ingest_pdfs.py` para ingesta batch (mismo flujo que el boton de Streamlit):

```powershell
python scripts\ingest_pdfs.py --pdf-dir .\books --domain literatura
python scripts\ingest_pdfs.py --pdf-dir .\books --domain literatura --force
```

### Observabilidad y salud

- Logging con **Loguru** a consola y a `logs/rag_books.log` con rotacion (5 MB) y retencion (10 dias).
- Prueba automatica de conexion a OpenAI al iniciar la app (recupera metadata del modelo configurado).

## Estructura del proyecto

- [app.py](app.py): interfaz Streamlit (configuracion, ingesta, listado de indice, chat con memoria y fuentes).
- [scripts/ingest_pdfs.py](scripts/ingest_pdfs.py): entrada CLI para ingesta batch.
- [src/rag_books/config.py](src/rag_books/config.py): carga de `.env` y dataclasses de configuracion (`DatabaseSettings`, `OpenAISettings`, `RagSettings`).
- [src/rag_books/db.py](src/rag_books/db.py): esquemas por dominio, tablas, indices vectoriales, busqueda kNN y busqueda diversificada por documento.
- [src/rag_books/embeddings.py](src/rag_books/embeddings.py): cliente OpenAI de embeddings con batching por items y por caracteres.
- [src/rag_books/llm.py](src/rag_books/llm.py): construccion del prompt y llamada a Chat Completions.
- [src/rag_books/retriever.py](src/rag_books/retriever.py): orquestacion de busqueda semantica y etiquetado de citas.
- [src/rag_books/ingest.py](src/rag_books/ingest.py): pipeline de ingesta (descubrir, extraer, chunk, embed, persistir).
- [src/rag_books/pdf_loader.py](src/rag_books/pdf_loader.py): listado, extraccion y hashing de PDFs.
- [src/rag_books/text_splitter.py](src/rag_books/text_splitter.py): chunking con solape y rango de paginas.
- [src/rag_books/domain_profiles.py](src/rag_books/domain_profiles.py): perfiles de prompt por dominio.
- [src/rag_books/openai_health.py](src/rag_books/openai_health.py): verificacion de conectividad con OpenAI.
- [src/rag_books/logging.py](src/rag_books/logging.py): configuracion centralizada de Loguru.

## Modelo de datos

Cada dominio es un **esquema PostgreSQL** independiente dentro de `RAG_DB`, con sus tablas `documents` y `chunks`. La extension `pgvector` vive en un esquema compartido (`extensions` por defecto), de modo que los tipos `vector(N)` y los operadores `vector_cosine_ops` se reutilizan entre dominios.

```mermaid
flowchart TB
    subgraph DB["RAG_DB (PostgreSQL)"]
        subgraph EXT["schema: extensions"]
            V["extension vector<br/>tipo vector(N)<br/>ops vector_cosine_ops"]
        end
        subgraph S1["schema: literatura"]
            D1[(documents)]
            C1[(chunks)]
            I1["HNSW / IVFFLAT<br/>sobre embedding"]
            D1 --- C1
            C1 -.-> I1
        end
        subgraph S2["schema: sistemas_gestion"]
            D2[(documents)]
            C2[(chunks)]
            I2["HNSW / IVFFLAT<br/>sobre embedding"]
            D2 --- C2
            C2 -.-> I2
        end
        subgraph S3["schema: &lt;RAG_DOMAIN&gt;"]
            D3[(documents)]
            C3[(chunks)]
            I3["HNSW / IVFFLAT<br/>sobre embedding"]
            D3 --- C3
            C3 -.-> I3
        end
    end
    V -.->|tipo vector| C1
    V -.->|tipo vector| C2
    V -.->|tipo vector| C3
```

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : "ON DELETE CASCADE"
    DOCUMENTS {
        uuid id PK
        text domain
        text source_path
        text file_name
        text title
        text author
        text content_hash
        jsonb metadata
        timestamptz created_at
    }
    CHUNKS {
        uuid id PK
        uuid document_id FK
        text domain
        int chunk_index
        int page_start
        int page_end
        text content
        int token_count
        vector embedding "dim = OPENAI_EMBEDDING_DIM"
        jsonb metadata
        timestamptz created_at
    }
```

Restricciones clave:

- `documents`: `UNIQUE (domain, source_path, content_hash)` habilita la deduplicacion por contenido.
- `chunks`: `UNIQUE (document_id, chunk_index)` permite reingesta idempotente con `ON CONFLICT ... DO UPDATE`.
- Indices secundarios: `documents_domain_idx`, `chunks_domain_idx` y el indice vectorial HNSW/IVFFLAT sobre `chunks.embedding`.

## Configuracion

`.env` contiene la configuracion local y esta ignorado por git. `.env.sample` contiene una plantilla segura con valores mock.

### Base de datos

- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: credenciales PostgreSQL.
- `DB_SSLMODE`: modo SSL de psycopg (por defecto `prefer`).
- `DB_EXTENSIONS_SCHEMA`: esquema donde reside `pgvector` (por defecto `extensions`).

### OpenAI

- `OPENAI_API_KEY`: clave de OpenAI. La app detecta placeholders comunes y los rechaza.
- `OPENAI_BASE_URL`: vacio para OpenAI oficial; si lo usas debe ser una URL completa (`https://...`).
- `OPENAI_CHAT_MODEL`: modelo para respuesta. Por defecto `gpt-5.5`.
- `OPENAI_EMBEDDING_MODEL` y `OPENAI_EMBEDDING_DIM`: modelo y dimension de embeddings. Por defecto `text-embedding-3-large` y `2000` (limite practico para HNSW/IVFFLAT de pgvector).
- `OPENAI_EMBEDDING_BATCH_SIZE`: maximo de chunks por llamada (por defecto `64`).
- `OPENAI_EMBEDDING_MAX_BATCH_CHARS`: presupuesto aproximado de caracteres por lote (por defecto `240000`).
- `OPENAI_TEMPERATURE`: solo se aplica a modelos que la soportan (no GPT-5).
- `OPENAI_REASONING_EFFORT`: `minimal`, `low`, `medium`, `high` para modelos compatibles (por defecto `medium`).
- `OPENAI_VERBOSITY`: `low`, `medium`, `high` para modelos compatibles (por defecto `high`).

### RAG

- `RAG_DOMAIN`: dominio logico y nombre del esquema PostgreSQL (identificador SQL valido).
- `RAG_PDF_DIR`: carpeta de entrada (los PDFs deben estar en la raiz, no en subcarpetas).
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`: tamano y solape de chunks en caracteres.
- `RAG_TOP_K`: chunks finales enviados al LLM.
- `RAG_CANDIDATE_K`: candidatos iniciales antes de diversificar.
- `RAG_MAX_CHUNKS_PER_DOCUMENT`: tope de chunks por documento durante la diversificacion.
- `RAG_MAX_CONTEXT_CHARS`: limite total de caracteres de contexto enviados al LLM.

## Instalacion con Conda

```powershell
conda env create -f environment.yml
conda activate rag-books
```

`requirements.txt` queda disponible si prefieres instalar las mismas dependencias con `pip`.

## Preparar base de datos

La app crea automaticamente el esquema del dominio, las tablas (`documents`, `chunks`) y los indices vectoriales. La extension `vector` debe existir o el usuario debe tener permiso para crearla:

```sql
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
```

Configura `DB_EXTENSIONS_SCHEMA=extensions` para que los indices encuentren las clases de operadores de pgvector (`vector_cosine_ops`).

## Ingestar PDFs

Coloca los PDFs en la raiz de la carpeta configurada (por ejemplo `books/`) y ejecuta:

```powershell
python scripts\ingest_pdfs.py --pdf-dir .\books --domain literatura
```

Para reingestar y reemplazar chunks existentes de un archivo:

```powershell
python scripts\ingest_pdfs.py --pdf-dir .\books --domain literatura --force
```

Tambien puedes ejecutar la ingesta desde el boton **Ingerir PDFs** del panel Indice en Streamlit.

## Ejecutar Streamlit

```powershell
streamlit run app.py
```

Desde la barra lateral puedes cambiar la carpeta de PDFs, el dominio, los parametros de recuperacion y chunking, probar la conexion con OpenAI y limpiar la conversacion.

En Windows tambien puedes abrir la app con doble clic en [run_app.bat](run_app.bat), que activa el entorno Conda `rag-books` y arranca Streamlit en el puerto 8501.

## Logs y prueba de API

La app usa `loguru` para mostrar paso a paso lo que ocurre en consola y guardar un archivo local en `logs/rag_books.log` con rotacion automatica (5 MB, retencion 10 dias).

Al abrir Streamlit se ejecuta una prueba de conexion a OpenAI con el modelo configurado en `OPENAI_CHAT_MODEL`. Tambien puedes repetirla en cualquier momento desde el boton **Probar OpenAI API** en la barra lateral.

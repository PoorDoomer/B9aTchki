# Developer Guide - OPUS Automated Pipeline

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure & Responsibilities](#file-structure--responsibilities)
4. [Configuration Management](#configuration-management)
5. [Database Schema](#database-schema)
6. [Common Modifications](#common-modifications)
   - [Changing Reclamation Table Column Names](#changing-reclamation-table-column-names)
   - [Modifying the Processing Flow](#modifying-the-processing-flow)
   - [Changing Similarity Thresholds](#changing-similarity-thresholds)
   - [Adding New Fields to Reclamations](#adding-new-fields-to-reclamations)
   - [Changing the ML Model](#changing-the-ml-model)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

The OPUS Automated Pipeline is a cross-lingual duplicate detection system for reclamation tickets. It uses:

- **LaBSE embeddings** (Language-agnostic BERT Sentence Embedding) for semantic similarity
- **PostgreSQL + pgvector** for vector similarity search
- **RabbitMQ** for event-driven processing
- **Python** for the processing logic

### Key Features

- Detects semantic duplicates across French and Arabic
- Real-time event-driven processing
- Contextual grouping (only matches within same user's reclamations)
- Configurable similarity thresholds
- Audit logging for all decisions

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│  Main App   │────▶│   RabbitMQ   │────▶│  ML Worker  │────▶│ PostgreSQL │
│ (Producer)  │     │    Queue     │     │ (Consumer)  │     │  + pgvector│
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
```

### Flow

1. **Producer** (`src/producer.py`) publishes events when new reclamations are created
2. **Worker** (`src/worker.py`) consumes events from RabbitMQ
3. **DuplicateDetector** (`src/duplicate_detector.py`) processes each reclamation:
   - Fetches reclamation from database
   - Preprocesses text (Arabic/French normalization)
   - Generates embeddings using LaBSE
   - Searches for similar reclamations using pgvector
   - Applies business rules (thresholds)
   - Updates status and logs results
4. **Database** stores reclamations, embeddings, and audit logs

---

## File Structure & Responsibilities

### Core Source Files (`src/`)

#### `src/config.py`
**Purpose**: Configuration management and environment variable loading

**Key Classes**:
- `PostgresConfig`: Database connection settings
- `RabbitMQConfig`: Message queue settings
- `MLConfig`: Machine learning model configuration
- `DuplicateDetectionConfig`: Similarity thresholds and time windows
- `Config`: Main configuration container

**What to modify**:
- Add new configuration sections
- Change default values
- Add new environment variables

**Example**: To add a new config section:
```python
@dataclass
class NewFeatureConfig:
    enabled: Optional[bool] = None
    
    def __post_init__(self):
        if self.enabled is None:
            self.enabled = os.getenv("NEW_FEATURE_ENABLED", "false").lower() == "true"

# Then add to Config class:
@dataclass
class Config:
    postgres: PostgresConfig
    rabbitmq: RabbitMQConfig
    ml: MLConfig
    detection: DuplicateDetectionConfig
    new_feature: NewFeatureConfig  # Add this
```

---

#### `src/database.py`
**Purpose**: Database connection pooling, CRUD operations, and vector similarity search

**Key Classes**:
- `DatabasePool`: Singleton connection pool manager
- `ReclamationRepository`: CRUD operations for reclamations table
- `EmbeddingRepository`: Vector embedding storage and similarity search
- `DuplicationLogRepository`: Audit log operations

**Key Data Classes**:
- `Reclamation`: Represents a reclamation record
- `SimilarityMatch`: Result from similarity search
- `DuplicationLog`: Audit log entry

**What to modify**:
- **Column names**: Update SQL queries in repository methods
- **Table names**: Change table references in queries
- **Query logic**: Modify similarity search queries
- **Data classes**: Update to match schema changes

**Example - Changing column names**:
```python
# In ReclamationRepository.get_by_id():
query = """
    SELECT id, user_id, message_libre, created_at, status
    FROM reclamations
    WHERE id = %s;
"""
# Change to:
query = """
    SELECT id, user_id, complaint_text, created_at, status
    FROM reclamations
    WHERE id = %s;
"""
# Also update the Reclamation dataclass:
@dataclass
class Reclamation:
    id: int
    user_id: int
    complaint_text: str  # Changed from message_libre
    created_at: datetime
    status: str
```

---

#### `src/preprocessing.py`
**Purpose**: Text normalization for Arabic and French text

**Key Functions**:
- `normalize_text()`: Main normalization pipeline
- `normalize_arabic()`: Arabic-specific normalization
- `normalize_french()`: French-specific normalization
- `clean_text()`: General text cleaning (HTML, URLs, emails)

**What to modify**:
- Add new normalization rules
- Change preprocessing steps
- Add support for new languages
- Modify cleaning rules

**Example - Adding new preprocessing step**:
```python
def remove_phone_numbers(text: str) -> str:
    """Remove phone numbers from text."""
    phone_pattern = re.compile(r'\b\d{10,}\b')
    return phone_pattern.sub(' ', text)

# Add to clean_text():
def clean_text(text: str) -> str:
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)  # Add this
    text = normalize_whitespace(text)
    return text
```

---

#### `src/embeddings.py`
**Purpose**: LaBSE model wrapper for generating embeddings

**Key Classes**:
- `EmbeddingModel`: Singleton wrapper for SentenceTransformer model

**Key Methods**:
- `encode()`: Generate embeddings (returns numpy array)
- `encode_to_list()`: Generate embeddings (returns Python list)
- `similarity()`: Compute similarity between two texts

**What to modify**:
- Change the model name (via config)
- Modify embedding parameters
- Add custom encoding logic

**Example - Using a different model**:
```python
# In .env file:
LABSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384  # Update dimension to match new model
```

---

#### `src/duplicate_detector.py`
**Purpose**: Core duplicate detection logic

**Key Classes**:
- `DuplicateDetector`: Main detection engine
- `DuplicateAction`: Enum for actions (AUTO_MARK_DUPLICATE, FLAG_FOR_REVIEW, NO_ACTION)
- `ReclamationStatus`: Enum for status values
- `DuplicateDetectionResult`: Result data class

**Key Methods**:
- `process_reclamation()`: Main processing pipeline
- `determine_action()`: Business rule logic based on similarity score
- `check_similarity()`: Utility for ad-hoc similarity checks

**What to modify**:
- Change business rules (thresholds, actions)
- Modify processing flow
- Add new status values
- Change how duplicates are determined

**Example - Changing business rules**:
```python
def determine_action(self, score: float) -> DuplicateAction:
    # Original:
    # if score >= 0.95: return AUTO_MARK_DUPLICATE
    # elif score >= 0.85: return FLAG_FOR_REVIEW
    
    # Modified (stricter):
    if score >= 0.98:  # Changed from 0.95
        return DuplicateAction.AUTO_MARK_DUPLICATE
    elif score >= 0.90:  # Changed from 0.85
        return DuplicateAction.FLAG_FOR_REVIEW
    else:
        return DuplicateAction.NO_ACTION
```

---

#### `src/producer.py`
**Purpose**: RabbitMQ message publisher

**Key Classes**:
- `RabbitMQProducer`: Publisher for reclamation events
- `ReclamationEvent`: Event data structure

**Key Methods**:
- `publish()`: Publish an event
- `publish_new_reclamation()`: Convenience method
- `publish_to_dlq()`: Publish to dead letter queue

**What to modify**:
- Change event structure
- Add new event types
- Modify queue names
- Add event metadata

**Example - Adding fields to event**:
```python
@dataclass
class ReclamationEvent:
    reclamation_id: int
    user_id: int
    event_type: str = "new_reclamation"
    priority: str = "NORMAL"  # Add this field
    source: str = "WEB"  # Add this field
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
```

---

#### `src/worker.py`
**Purpose**: RabbitMQ consumer worker

**Key Classes**:
- `RabbitMQWorker`: Consumer for processing events

**Key Methods**:
- `start()`: Start consuming messages
- `_process_message()`: Process a single message
- `process_one()`: Process one message and return (for testing)

**What to modify**:
- Change message processing logic
- Add error handling
- Modify retry logic
- Add custom callbacks

**Example - Adding custom processing**:
```python
def _process_message(self, channel, method, properties, body):
    try:
        event = ReclamationEvent.from_json(body.decode('utf-8'))
        
        # Add custom logic before processing
        if event.priority == "HIGH":
            logger.info(f"High priority reclamation {event.reclamation_id}")
        
        detector = self._get_detector()
        result = detector.process_reclamation(event.reclamation_id)
        
        # Add custom logic after processing
        if result.is_duplicate:
            # Send notification, etc.
            pass
        
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        # Error handling
        pass
```

---

### Configuration Files

#### `.env` / `env.example`
**Purpose**: Environment variable definitions

**What to modify**:
- Database connection settings
- RabbitMQ settings
- ML model configuration
- Threshold values

**See**: [Configuration Management](#configuration-management)

---

#### `docker-compose.yml`
**Purpose**: Docker infrastructure setup

**What to modify**:
- Database credentials
- Port mappings
- Volume mounts
- Service configurations

---

#### `migrations/001_initial_schema.sql`
**Purpose**: Database schema definition

**What to modify**:
- Table structures
- Column definitions
- Indexes
- Constraints

**See**: [Database Schema](#database-schema)

---

## Configuration Management

### Environment Variables

All configuration is loaded from environment variables (or `.env` file). See `env.example` for all available variables.

### PostgreSQL Configuration

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=dedup_user
POSTGRES_PASSWORD=dedup_password
POSTGRES_DB=dedup_db
```

### RabbitMQ Configuration

```bash
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=dedup_user
RABBITMQ_PASSWORD=dedup_password
RABBITMQ_QUEUE=reclamation_processing
RABBITMQ_DLQ=dead_letter_queue
```

### ML Configuration

```bash
LABSE_MODEL=sentence-transformers/LaBSE
EMBEDDING_DIMENSION=768
```

### Duplicate Detection Thresholds

```bash
THRESHOLD_AUTO_DUPLICATE=0.95  # Auto-mark as duplicate if score >= this
THRESHOLD_REVIEW=0.85          # Flag for review if score >= this
TIME_WINDOW_DAYS=7             # Look back N days for duplicates
```

### Loading Configuration in Code

```python
from src.config import get_config

config = get_config()
print(config.postgres.host)
print(config.detection.threshold_auto_duplicate)
```

---

## Database Schema

### Tables

#### `reclamations`
Main table storing reclamation tickets.

**Columns**:
- `id` (BIGSERIAL PRIMARY KEY)
- `user_id` (BIGINT NOT NULL) - Grouping key
- `message_libre` (TEXT NOT NULL) - Raw text
- `created_at` (TIMESTAMPTZ DEFAULT NOW())
- `status` (VARCHAR(50) DEFAULT 'PENDING')

**Status Values**:
- `PENDING`: New, not processed
- `DUPLICATE`: Auto-marked as duplicate
- `POTENTIAL_DUPLICATE`: Flagged for review
- `PROCESSED`: Processed successfully

#### `reclamation_embeddings`
Vector store for embeddings.

**Columns**:
- `reclamation_id` (BIGINT PRIMARY KEY, FK to reclamations.id)
- `embedding` (vector(768)) - LaBSE embedding vector

**Index**: HNSW index for fast similarity search

#### `duplication_logs`
Audit log for duplicate detection decisions.

**Columns**:
- `id` (BIGSERIAL PRIMARY KEY)
- `source_reclamation_id` (BIGINT, FK to reclamations.id)
- `matched_reclamation_id` (BIGINT, FK to reclamations.id)
- `similarity_score` (DECIMAL(5, 4))
- `action` (VARCHAR(50)) - AUTO_MARK_DUPLICATE or FLAG_FOR_REVIEW
- `detected_at` (TIMESTAMPTZ DEFAULT NOW())

---

## Common Modifications

### Changing Reclamation Table Column Names

If your database uses different column names, you need to update multiple files:

#### Step 1: Update Database Schema (`migrations/001_initial_schema.sql`)

```sql
-- Change message_libre to complaint_text
CREATE TABLE IF NOT EXISTS reclamations (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    complaint_text TEXT NOT NULL,  -- Changed from message_libre
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'PENDING'
);
```

#### Step 2: Update Data Class (`src/database.py`)

```python
@dataclass
class Reclamation:
    id: int
    user_id: int
    complaint_text: str  # Changed from message_libre
    created_at: datetime
    status: str
```

#### Step 3: Update Repository Queries (`src/database.py`)

Update all SQL queries in `ReclamationRepository`:

```python
# In create():
query = """
    INSERT INTO reclamations (user_id, complaint_text, status)
    VALUES (%s, %s, %s)
    RETURNING id;
"""

# In get_by_id():
query = """
    SELECT id, user_id, complaint_text, created_at, status
    FROM reclamations
    WHERE id = %s;
"""
```

#### Step 4: Update Duplicate Detector (`src/duplicate_detector.py`)

```python
# In process_reclamation():
normalized_text = normalize_text(reclamation.complaint_text)  # Changed from message_libre
```

#### Step 5: Update Embedding Repository (`src/database.py`)

```python
# In find_similar():
query = """
    SELECT 
        r.id as reclamation_id,
        1 - (e.embedding <=> %s::vector) as score,
        r.complaint_text,  # Changed from message_libre
        r.status
    FROM reclamations r
    JOIN reclamation_embeddings e ON r.id = e.reclamation_id
    ...
"""
```

#### Step 6: Update SimilarityMatch Data Class (`src/database.py`)

```python
@dataclass
class SimilarityMatch:
    reclamation_id: int
    score: float
    complaint_text: Optional[str] = None  # Changed from message_libre
    status: Optional[str] = None
```

#### Step 7: Run Database Migration

```bash
# Connect to database and run migration
psql -h localhost -U dedup_user -d dedup_db -f migrations/002_rename_message_libre.sql
```

#### Step 8: Update Tests

Update all test files that reference `message_libre`:
- `tests/test_database.py`
- `tests/test_duplicate_detector.py`
- `tests/test_integration.py`

---

### Modifying the Processing Flow

#### Adding a New Processing Step

**Example**: Add email notification when duplicate is detected.

**Step 1**: Create notification module (`src/notifications.py`):

```python
import logging
from src.duplicate_detector import DuplicateDetectionResult

logger = logging.getLogger(__name__)

def send_duplicate_notification(result: DuplicateDetectionResult):
    """Send email notification for duplicate detection."""
    if result.is_duplicate:
        logger.info(f"Sending notification for duplicate: {result.reclamation_id}")
        # Add email sending logic here
```

**Step 2**: Update Worker (`src/worker.py`):

```python
from src.notifications import send_duplicate_notification

def _process_message(self, channel, method, properties, body):
    # ... existing code ...
    
    result = detector.process_reclamation(event.reclamation_id)
    
    # Add notification step
    if result.is_duplicate:
        send_duplicate_notification(result)
    
    # ... rest of code ...
```

#### Changing the Order of Processing Steps

**Example**: Generate embedding before preprocessing.

**In `src/duplicate_detector.py`**:

```python
def process_reclamation(self, reclamation_id: int) -> DuplicateDetectionResult:
    reclamation = self._reclamation_repo.get_by_id(reclamation_id)
    
    # Original order:
    # 1. Preprocess
    # 2. Generate embedding
    
    # New order:
    # 1. Generate embedding (raw text)
    model = get_embedding_model()
    raw_embedding = model.encode_to_list(reclamation.message_libre)
    
    # 2. Preprocess (for search)
    normalized_text = normalize_text(reclamation.message_libre)
    normalized_embedding = model.encode_to_list(normalized_text)
    
    # Store normalized embedding
    self._embedding_repo.save_embedding(reclamation_id, normalized_embedding)
    
    # Continue with search...
```

---

### Changing Similarity Thresholds

#### Method 1: Environment Variables (Recommended)

Update `.env` file:

```bash
THRESHOLD_AUTO_DUPLICATE=0.98  # Stricter
THRESHOLD_REVIEW=0.90          # Stricter
TIME_WINDOW_DAYS=14            # Look back 2 weeks instead of 1
```

Restart the worker for changes to take effect.

#### Method 2: Code Changes

**In `src/duplicate_detector.py`**:

```python
def determine_action(self, score: float) -> DuplicateAction:
    # Custom thresholds
    if score >= 0.98:  # Changed from config value
        return DuplicateAction.AUTO_MARK_DUPLICATE
    elif score >= 0.90:
        return DuplicateAction.FLAG_FOR_REVIEW
    else:
        return DuplicateAction.NO_ACTION
```

**In `src/duplicate_detector.py.process_reclamation()`**:

```python
matches = self._embedding_repo.find_similar(
    embedding=embedding,
    user_id=reclamation.user_id,
    exclude_id=reclamation_id,
    min_score=0.90,  # Changed from self.threshold_review
    time_window_days=14,  # Changed from self.time_window_days
    limit=1
)
```

---

### Adding New Fields to Reclamations

#### Step 1: Update Database Schema

Create new migration file (`migrations/002_add_priority.sql`):

```sql
ALTER TABLE reclamations 
ADD COLUMN priority VARCHAR(20) DEFAULT 'NORMAL';

ALTER TABLE reclamations
ADD CONSTRAINT chk_priority CHECK (priority IN ('LOW', 'NORMAL', 'HIGH', 'URGENT'));
```

#### Step 2: Update Data Class

**In `src/database.py`**:

```python
@dataclass
class Reclamation:
    id: int
    user_id: int
    message_libre: str
    created_at: datetime
    status: str
    priority: str = "NORMAL"  # Add this
```

#### Step 3: Update Repository Methods

**In `src/database.py.ReclamationRepository`**:

```python
def create(self, user_id: int, message_libre: str, status: str = "PENDING", priority: str = "NORMAL") -> int:
    query = """
        INSERT INTO reclamations (user_id, message_libre, status, priority)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """
    with self._pool.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id, message_libre, status, priority))
            # ...

def get_by_id(self, reclamation_id: int) -> Optional[Reclamation]:
    query = """
        SELECT id, user_id, message_libre, created_at, status, priority
        FROM reclamations
        WHERE id = %s;
    """
    # ...
```

#### Step 4: Use in Processing Logic

**In `src/duplicate_detector.py`**:

```python
def process_reclamation(self, reclamation_id: int) -> DuplicateDetectionResult:
    reclamation = self._reclamation_repo.get_by_id(reclamation_id)
    
    # Use priority in logic
    if reclamation.priority == "URGENT":
        # Skip duplicate detection for urgent tickets
        return DuplicateDetectionResult(...)
    
    # ... rest of processing ...
```

---

### Changing the ML Model

#### Step 1: Update Configuration

**In `.env`**:

```bash
LABSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384  # Must match model dimension
```

#### Step 2: Update Database Schema

If changing dimension, you need to recreate the embeddings table:

```sql
-- Drop old table
DROP TABLE IF EXISTS reclamation_embeddings CASCADE;

-- Create with new dimension
CREATE TABLE reclamation_embeddings (
    reclamation_id BIGINT PRIMARY KEY REFERENCES reclamations(id) ON DELETE CASCADE,
    embedding vector(384)  -- New dimension
);

-- Recreate index
CREATE INDEX idx_embeddings_hnsw ON reclamation_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

#### Step 3: Regenerate Embeddings

Create a script to regenerate all embeddings:

```python
# scripts/regenerate_embeddings.py
from src.database import get_reclamation_repo, get_embedding_repo
from src.duplicate_detector import DuplicateDetector

repo = get_reclamation_repo()
embedding_repo = get_embedding_repo()
detector = DuplicateDetector()

# Get all reclamations
# Regenerate embeddings
# (Implementation depends on your needs)
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_database.py -v

# Run specific test
python -m pytest tests/test_database.py::test_create_reclamation -v
```

### Test Structure

- `tests/test_config.py`: Configuration tests
- `tests/test_database.py`: Database operations tests
- `tests/test_preprocessing.py`: Text preprocessing tests
- `tests/test_embeddings.py`: Embedding generation tests
- `tests/test_duplicate_detector.py`: Duplicate detection logic tests
- `tests/test_producer.py`: RabbitMQ producer tests
- `tests/test_worker.py`: Worker consumer tests
- `tests/test_integration.py`: End-to-end integration tests

### Writing New Tests

**Example**: Test for new priority field

```python
# tests/test_database.py
def test_create_reclamation_with_priority():
    repo = get_reclamation_repo()
    rec_id = repo.create(
        user_id=1,
        message_libre="Test complaint",
        status="PENDING",
        priority="HIGH"
    )
    
    reclamation = repo.get_by_id(rec_id)
    assert reclamation.priority == "HIGH"
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

**Error**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
- Check PostgreSQL is running: `docker ps`
- Verify connection settings in `.env`
- Check firewall/network settings

#### 2. RabbitMQ Connection Errors

**Error**: `AMQPConnectionError`

**Solution**:
- Check RabbitMQ is running: `docker ps`
- Verify credentials in `.env`
- Check RabbitMQ management UI: http://localhost:15672

#### 3. Model Loading Errors

**Error**: `ImportError: sentence-transformers not installed`

**Solution**:
```bash
pip install sentence-transformers
```

#### 4. Vector Dimension Mismatch

**Error**: `dimension mismatch: vector(768) vs vector(384)`

**Solution**:
- Ensure `EMBEDDING_DIMENSION` matches model dimension
- Update database schema if needed
- Regenerate embeddings

#### 5. Column Not Found Errors

**Error**: `column "message_libre" does not exist`

**Solution**:
- Check database schema matches code
- Run migrations
- Verify column names in all SQL queries

### Debugging Tips

1. **Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check Queue Size**:
```python
from src.worker import RabbitMQWorker
worker = RabbitMQWorker()
print(f"Queue size: {worker.get_queue_size()}")
```

3. **Test Individual Components**:
```python
# Test preprocessing
from src.preprocessing import normalize_text
print(normalize_text("Test text"))

# Test embeddings
from src.embeddings import encode_text
embedding = encode_text("Test text")
print(f"Embedding dimension: {len(embedding)}")

# Test duplicate detection
from src.duplicate_detector import detect_duplicates
result = detect_duplicates(reclamation_id=123)
print(result)
```

---

## Best Practices

1. **Always update tests** when modifying code
2. **Use migrations** for database schema changes
3. **Update documentation** when adding features
4. **Use environment variables** for configuration
5. **Follow the existing code style** and patterns
6. **Test locally** before deploying
7. **Monitor logs** for errors and performance

---

## Additional Resources

- [PostgreSQL pgvector Documentation](https://github.com/pgvector/pgvector)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [LaBSE Model](https://huggingface.co/sentence-transformers/LaBSE)
- [Sentence Transformers Documentation](https://www.sbert.net/)

---

## Support

For questions or issues, refer to:
- Project README.md
- Test files for usage examples
- Code comments and docstrings

---

**Last Updated**: 2025-01-27



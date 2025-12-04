# Pipeline kat detecter wach derti double reclamation ( chki chki hh)

Ach kan 3ni : Automated semantic de-duplication pipeline for reclamation tickets in French and Arabic using LaBSE embeddings and pgvector similarity search.
3lach LaBSE : 7ent tout simplement it is one of the best performing language agnostic type of embedding system
## Features

- **Cross-Lingual Support**: Detects semantic duplicates across French and Arabic using Google's LaBSE model
- **Real-time Processing**: Event-driven architecture with RabbitMQ message queue
- **Contextual Grouping**: Duplicates are only matched within the same user's reclamations
- **Configurable Thresholds**: 
  - `>= 0.95`: Auto-mark as duplicate
  - `0.85 - 0.95`: Flag for manual review
  - `< 0.85`: Unique (no action)

## L'Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│  Main App   │────▶│   RabbitMQ   │────▶│  ML Worker  │────▶│ PostgreSQL │
│ (Producer)  │     │    Queue     │     │ (Consumer)  │     │  + pgvector│
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
```

## Project Structure

```
├── docker-compose.yml          # PostgreSQL + pgvector + RabbitMQ
├── requirements.txt            # Python dependencies
├── migrations/
│   └── 001_initial_schema.sql  # Database schema with vector support
├── src/
│   ├── config.py               # Configuration management
│   ├── preprocessing.py        # Arabic/French text normalization
│   ├── embeddings.py           # LaBSE model wrapper (singleton)
│   ├── database.py             # PostgreSQL connection & queries
│   ├── duplicate_detector.py   # Core similarity logic
│   ├── producer.py             # RabbitMQ message publisher
│   └── worker.py               # RabbitMQ consumer worker
├── tests/                      # Unit and integration tests (150 tests)
└── scripts/
    └── generate_mock_data.py   # Mock data generator
```

## 2intila9a Sari3aaaaa

### 1. Start Infrastructure

```bash
docker-compose up -d
```

Hdchi will starts:
- PostgreSQL 15 avec le  pgvector extension (port 5432)
- RabbitMQ with management UI (ports 5672, 15672)

### 2. Installi Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Start the Worker

```bash
python -m src.worker
```

### 5. Publish Events

```python
from src.producer import publish_reclamation_event

# When a new reclamation is created
publish_reclamation_event(reclamation_id=123, user_id=10)
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
| `POSTGRES_USER` | dedup_user | Database user |
| `POSTGRES_PASSWORD` | dedup_password | Database password |
| `POSTGRES_DB` | dedup_db | Database name |
| `RABBITMQ_HOST` | localhost | RabbitMQ host |
| `RABBITMQ_PORT` | 5672 | RabbitMQ port |
| `THRESHOLD_AUTO_DUPLICATE` | 0.95 | Auto-duplicate threshold |
| `THRESHOLD_REVIEW` | 0.85 | Review flag threshold |
| `TIME_WINDOW_DAYS` | 7 | Days to look back for duplicates |

## Text Preprocessing

### French
- Lowercase conversion
- HTML tag removal
- URL/email removal
- Whitespace normalization

### Arabic
- Diacritics (Harakat) removal
- Tatweel (Kashida) removal
- Alif normalization (أ, إ, آ → ا)
- Ya normalization (ي → ى)

## API Usage

### Direct Detection

```python
from src.duplicate_detector import detect_duplicates

result = detect_duplicates(reclamation_id=123)
print(f"Is duplicate: {result.is_duplicate}")
print(f"Action: {result.action}")
print(f"Matched ID: {result.matched_id}")
print(f"Score: {result.similarity_score}")
```

### Similarity Check

```python
from src.duplicate_detector import get_duplicate_detector

detector = get_duplicate_detector()
score, action = detector.check_similarity(
    "Ma connexion internet ne fonctionne pas",
    "L'internet est coupé"
)
```

## Database Schema

### Tables

- **reclamations**: Main reclamation records
- **reclamation_embeddings**: 768-dimensional LaBSE vectors with HNSW index
- **duplication_logs**: Audit trail of detection decisions

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test module
python -m pytest tests/test_preprocessing.py -v
```

## Mock Data Generation

```bash
# Generate 10 users with 5 reclamations each
python scripts/generate_mock_data.py --users 10 --reclamations 5

# Without duplicates
python scripts/generate_mock_data.py --no-duplicates
```

## License

Internal use only - ALEXSYS SOLUTIONS



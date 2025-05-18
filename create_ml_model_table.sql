-- Script per creare la tabella ml_model se non esiste
CREATE TABLE IF NOT EXISTS ml_model (
    id INTEGER PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    parameters JSON,
    metrics JSON,
    model_path VARCHAR(256),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER NOT NULL,
    dataset_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES user(id),
    FOREIGN KEY (dataset_id) REFERENCES dataset(id)
);
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    sex CHAR(1),
    mrn TEXT UNIQUE,
    diagnosis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    patient_id TEXT REFERENCES patients(id),
    features JSONB NOT NULL,
    probability FLOAT,
    risk_level TEXT,
    confidence TEXT,
    trajectory JSONB,
    shap_values JSONB,
    causal_graph JSONB,
    report_sections JSONB,
    disease_classification JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_patient ON analyses(patient_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

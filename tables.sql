-- ******************************* PRODUCTION TABLES *******************************
CREATE TABLE organizations (
  org_id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE brands (
  brand_id UUID PRIMARY KEY,
  org_id UUID REFERENCES organizations(org_id),
  name TEXT NOT NULL,
  industry TEXT,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE brand_assets (
  asset_id UUID PRIMARY KEY,
  brand_id UUID REFERENCES brands(brand_id),
  asset_type TEXT CHECK (asset_type IN ('copy','guideline','website')),
  raw_text TEXT,
  source TEXT,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE brand_chunks (
  chunk_id UUID PRIMARY KEY,
  asset_id UUID REFERENCES brand_assets(asset_id),
  brand_id UUID REFERENCES brands(brand_id),
  vector_type TEXT CHECK (vector_type IN ('brand_voice','strategy','performance')),
  content TEXT,
  token_count INT,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE embeddings (
  embedding_id UUID PRIMARY KEY,
  chunk_id UUID REFERENCES brand_chunks(chunk_id),
  brand_id UUID REFERENCES brands(brand_id),
  vector_type TEXT,
  namespace TEXT,
  model TEXT,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE brand_insights (
  insight_id UUID PRIMARY KEY,
  brand_id UUID REFERENCES brands(brand_id),
  insight_type TEXT,
  insight_json JSONB,
  confidence FLOAT,
  created_at TIMESTAMP DEFAULT now()
);


CREATE TABLE feedback (
  feedback_id UUID PRIMARY KEY,
  brand_id UUID REFERENCES brands(brand_id),
  output_text TEXT,
  edited_text TEXT,
  rating INT CHECK (rating BETWEEN 1 AND 5),
  created_at TIMESTAMP DEFAULT now()
);


-- ************************** DROP COMMANDS, IF REQUIRED **********
-- drop table organizations;
-- drop table brands;
-- drop table brand_assets;
-- drop table brand_chunks;
-- drop table embeddings;
-- drop table brand_insights;
-- drop table feedback;

-- Schema bootstrap for the affective memory layer.
-- Mem0 manages its OWN vector collection tables; these are the prototype's
-- application-level tables that the pre-call lookup and post-call worker read/write.

CREATE EXTENSION IF NOT EXISTS vector;

-- The latest known affective state per (tenant, user). The pre-call hook reads this
-- to pick a prosody profile; the post-call worker upserts it after every call.
CREATE TABLE IF NOT EXISTS affective_state (
    tenant_id     TEXT        NOT NULL,
    user_id       TEXT        NOT NULL,
    emotion       TEXT        NOT NULL DEFAULT 'neutral',  -- frustrated|angry|happy|neutral|sad...
    valence       REAL        NOT NULL DEFAULT 0.0,        -- -1..1
    arousal       REAL        NOT NULL DEFAULT 0.0,        --  0..1
    confidence    REAL        NOT NULL DEFAULT 0.0,        --  0..1
    features      JSONB       NOT NULL DEFAULT '{}'::jsonb,-- low-level numeric features
    paralinguistics JSONB     NOT NULL DEFAULT '{}'::jsonb,-- rich SenseVoice+openSMILE profile
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

-- An audit log of every contradiction-engine decision. Mem0's resolver is
-- probabilistic, so we keep a deterministic trail of what changed and why.
CREATE TABLE IF NOT EXISTS assertion_audit (
    id            BIGSERIAL   PRIMARY KEY,
    tenant_id     TEXT        NOT NULL,
    user_id       TEXT        NOT NULL,
    call_sid      TEXT,
    op            TEXT        NOT NULL,                    -- ADD|UPDATE|DELETE|NOOP
    new_fact      TEXT        NOT NULL,
    superseded    TEXT,                                    -- prior memory text, if any
    evidence      TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assertion_audit_user
    ON assertion_audit (tenant_id, user_id, created_at DESC);
